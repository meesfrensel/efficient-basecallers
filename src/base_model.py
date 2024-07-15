"""
Contains a base model implementation.

Adapted from https://github.com/marcpaga/basecalling_architectures/blob/5db4957496079d19deacb01c9f4f4957f7257f49/src/classes.py
"""

from abc import abstractmethod
import torch
from torch import nn
from fast_ctc_decode import crf_greedy_search, crf_beam_search

from constants import CTC_BLANK, BASES_CRF
from constants import CRF_STATE_LEN, CRF_BIAS, CRF_SCALE, CRF_BLANK_SCORE, CRF_N_BASE

from evaluation import alignment_accuracy

from layers.bonito import CTC_CRF, BonitoLinearCRFDecoder
from layers.masked_lstm import change_lstm_cell

class BaseModel(nn.Module):
    """Abstract class for basecaller models

    It contains some basic methods: train, validate, predict, ctc_decode...
    Since most models follow a similar style.
    """
    
    def __init__(self, device, dataloader_train, dataloader_validation, 
                 optimizer = None, schedulers = dict(), criterions = dict(), clipping_value = 2, scaler = None, use_amp = False):
        super(BaseModel, self).__init__()
        
        self.device = device
        
        # data
        self.dataloader_train = dataloader_train
        self.dataloader_validation = dataloader_validation

        # optimization
        self.optimizer = optimizer
        self.schedulers = schedulers
        self.criterions = criterions
        self.clipping_value = clipping_value
        self.scaler = scaler
        if self.scaler is not None:
            self.use_amp = True
        else:
            self.use_amp = use_amp
        
        self.init_weights()
        self.stride = self.get_stride()
        self.dummy_batch = None
        
    @abstractmethod
    def forward(self, batch):
        """Forward through the network
        """
        raise NotImplementedError()
    
    def train_step(self, batch):
        """Train a step with a batch of data
        
        Args:
            batch (dict): dict with keys 'x' (batch, len) 
                                         'y' (batch, len)
        """
        
        self.train()
        x = batch['x'].to(self.device)
        x = x.unsqueeze(1) # add channels dimension
        y = batch['y'].to(self.device)
        
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            p = self.forward(x) # forward through the network
            loss, losses = self.calculate_loss(y, p)

        self.optimize(loss)
        
        return losses, p
    
    def validation_step(self, batch):
        """Predicts a single batch of data
        Args:
            batch (dict): dict filled with tensors of input and output
        """
        
        self.eval()
        with torch.no_grad():
            x = batch['x'].to(self.device)
            x = x.unsqueeze(1) # add channels dimension
            y = batch['y'].to(self.device)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                p = self.forward(x) # forward through the network
                _, losses = self.calculate_loss(y, p)
            
        return losses, p
    
    def predict_step(self, batch):
        """
        Args:
            batch (dict) dict fill with tensor just for prediction
        """
        self.eval()
        with torch.no_grad():
            x = batch['x'].to(self.device)
            x = x.unsqueeze(1)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                p = self.forward(x)
            
        return p
    
    def optimize(self, loss):
        """Optimizes the model by calculating the loss and doing backpropagation
        
        Args:
            loss (float): calculated loss that can be backpropagated
        """
        
        if self.scaler is not None:
            self.scaler.scale(loss).backward()

            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.clipping_value)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        else:
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.clipping_value)
            self.optimizer.step()
            
        for scheduler in self.schedulers.values():
            if scheduler:
                scheduler.step()
            
        return None

    def evaluate(self, batch, predictions):
        """Evaluate the predictions by calculating the accuracy
        
        Args:
            batch (dict): dict with tensor with [batch, len] in key 'y'
            predictions (list): list of predicted sequences as strings
        """
        y = batch['y'].cpu().numpy()
        y_list = self.dataloader_train.dataset.encoded_array_to_list_strings(y)
        accs = list()
        for i, sample in enumerate(y_list):
            accs.append(alignment_accuracy(sample, predictions[i]))
            
        return {'metric.accuracy': accs}
    
    def init_weights(self):
        """Initialize weights from uniform distribution
        """
        for name, param in self.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)

    def count_parameters(self):
        """Count trainable parameters in model
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save(self, checkpoint_file):
        """Save the model state
        """
        if self.scaler is not None:
            scaler_dict = self.scaler.state_dict()
        else:
            scaler_dict = None
            
        save_dict = {'model_state': self.state_dict(), 
                     'optimizer_state': self.optimizer.state_dict(),
                     'scaler': scaler_dict}

        for k, v in self.schedulers.items():
            save_dict[k + '_state'] = v.state_dict()
        torch.save(save_dict, checkpoint_file)
    
    def load(self, checkpoint_file, initialize_lazy = True):
        """Load a model state from a checkpoint file

        Args:
            checkpoint_file (str): file were checkpoints are saved
            initialize_lazy (bool): to do a forward step before loading model,
                this solves the problem for layers that are initialized lazyly
        """

        if initialize_lazy:
            if self.dummy_batch is None:
                dummy_batch = {'x': torch.randn([16, 1000], device = self.device)}
            else:
                dummy_batch = self.dummy_batch
            self.predict_step(dummy_batch)

        checkpoints = torch.load(checkpoint_file)
        self.load_state_dict(checkpoints['model_state'], strict=False)

        # # For old pytorch versions that do not support register_load_state_dict_post_hook (see masked_lstm.py)
        # for name, m in self.named_modules():
        #     if 'encoder.' in name and 'lstm' not in name:
        #         change_lstm_cell(m)

        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoints['optimizer_state'])
        if self.scaler is not None:
            self.optimizer.load_state_dict(checkpoints['scaler'])
        if 'lr_scheduler' in list(self.schedulers.keys()):
            self.schedulers['lr_scheduler'].load_state_dict(checkpoints['lr_scheduler_state'])

        
    def get_stride(self):
        """Gives the total stride of the model
        """
        return None
        
    @abstractmethod
    def load_default_configuration(self, default_all = False):
        """Method to load default model configuration
        """
        raise NotImplementedError()

class BaseModelCRF(BaseModel):
    
    def __init__(self, state_len = 4, alphabet = BASES_CRF, *args, **kwargs):
        """
        Args:
            state_len (int): k-mer length for the states
            alphabet (str): bases available for states, defaults 'NACGT'
        """
        super(BaseModelCRF, self).__init__(*args, **kwargs)

        self.alphabet = alphabet
        self.state_len = alphabet
        self.seqdist = CTC_CRF(state_len = state_len, alphabet = alphabet)
        self.criterions = {'crf': self.seqdist.ctc_loss}
        
    def decode(self, p, greedy = True, *args, **kwargs):
        """Decode the predictions
        
        Args:
            p (tensor): tensor with the predictions with shape [timesteps, batch, classes]
            greedy (bool): whether to decode using a greedy approach
        Returns:
            A (list) with the decoded strings
        """
        if greedy:
            return self.decode_crf_greedy(p, *args, **kwargs)
        else:
            return self.decode_crf_beamsearch(p, *args, **kwargs)

    def compute_scores(self, probs, use_fastctc = False):
        """
        Args:
            probs (cuda tensor): [length, batch, channels]
            use_fastctc (bool)
        """
        if use_fastctc:
            scores = probs.cuda().to(torch.float32)
            betas = self.seqdist.backward_scores(scores.to(torch.float32))
            trans, init = self.seqdist.compute_transition_probs(scores, betas)
            trans = trans.to(torch.float32).transpose(0, 1)
            init = init.to(torch.float32).unsqueeze(1)
            return (trans, init)
        else:
            scores = self.seqdist.posteriors(probs.cuda().to(torch.float32)) + 1e-8
            tracebacks = self.seqdist.viterbi(scores.log()).to(torch.int16).T
            return tracebacks

    def _decode_crf_greedy_fastctc(self, tracebacks, init, qstring, qscale, qbias, return_path):
        """
        Args:
            tracebacks (np.array): [len, states, bases]
            init (np.array): [states]
            qstring (bool)
            qscale (float)
            qbias (float)
            return_path (bool)
        """

        seq, path = crf_greedy_search(
            network_output = tracebacks, 
            init_state = init, 
            alphabet = BASES_CRF, 
            qstring = qstring, 
            qscale = qscale, 
            qbias = qbias
        )
        if return_path:
            return seq, path
        else:
            return seq
    
    def decode_crf_greedy(self, probs, use_fastctc = False, qstring = False, qscale = 1.0, qbias = 1.0, return_path = False, *args, **kwargs):
        """Predict the sequences using a greedy approach
        
        Args:
            probs (tensor): tensor with scores in shape [timesteps, batch, classes]
        Returns:
            A (list) with the decoded strings
        """

        if use_fastctc:
            tracebacks, init = self.compute_scores(probs, use_fastctc)
            return self._decode_crf_greedy_fastctc(tracebacks, init, qstring, qscale, qbias, return_path)
        
        else:
            return [self.seqdist.path_to_str(y) for y in self.compute_scores(probs, use_fastctc).cpu().numpy()]

    def _decode_crf_beamsearch_fastctc(self, tracebacks, init, beam_size, beam_cut_threshold, return_path):
        """
        Args
            tracebacks (np.array): [len, states, bases]
            init (np.array): [states]
            beam_size (int)
            beam_cut_threshold (float)
            return_path (bool)
        """
        seq, path = crf_beam_search(
            network_output = tracebacks, 
            init_state = init, 
            alphabet = BASES_CRF, 
            beam_size = beam_size,
            beam_cut_threshold = beam_cut_threshold
        )
        if return_path:
            return seq, path
        else:
            return seq

    def decode_crf_beamsearch(self, probs, beam_size = 5, beam_cut_threshold = 0.1, return_path = False, *args, **kwargs):
        """Predict the sequences using a beam search
        
        Args:
            probs (tensor): tensor with scores in shape [timesteps, batch, classes]
        Returns:
            A (list) with the decoded strings
        """

        tracebacks, init = self.compute_scores(probs, use_fastctc = True)
        return self._decode_crf_beamsearch_fastctc(tracebacks, init, beam_size, beam_cut_threshold, return_path)

    def calculate_loss(self, y, p):
        """Calculates the losses for each criterion
        
        Args:
            y (tensor): tensor with labels [batch, len]
            p (tensor): tensor with predictions [len, batch, channels]
            
        Returns:
            loss (tensor): weighted sum of losses
            losses (dict): with detached values for each loss, the weighed sum is named
                global_loss
        """
        
        loss = self.calculate_crf_loss(y, p)
        losses = {'loss.global': loss.item(), 'loss.crf': loss.item()}

        return loss, losses

    def calculate_crf_loss(self, y, p):
        """Calculates the losses for each criterion
        
        Args:
            y (tensor): tensor with labels [batch, len]
            p (tensor): tensor with predictions [len, batch, channels]
            
        Returns:
            loss (tensor): weighted sum of losses
            losses (dict): with detached values for each loss, the weighed sum is named
                global_loss
        """

        y_len = torch.sum(y != CTC_BLANK, axis = 1).to(self.device)
        loss = self.criterions['crf'](scores = p, 
                                      targets = y, 
                                      target_lengths = y_len, 
                                      loss_clip = 10, 
                                      reduction='mean', 
                                      normalise_scores=True)
        return loss

    def build_decoder(self, encoder_output_size):
        decoder = BonitoLinearCRFDecoder(
            insize = encoder_output_size, 
            n_base = CRF_N_BASE, 
            state_len = CRF_STATE_LEN, 
            bias=CRF_BIAS, 
            scale= CRF_SCALE, 
            blank_score= CRF_BLANK_SCORE
        )

        return decoder
