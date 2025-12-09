from abc import ABC, abstractmethod
from typing import Optional
from stable_baselines3 import PPO

class ExpertPolicy(ABC):

    @property
    @abstractmethod
    def beta(self):
        return self.beta

    @abstractmethod
    def __call__(self, obs):
        pass

class ActorCriticExpertPolicy(ExpertPolicy):

    def __init__(self, beta: float, path: Optional[str] = None, ppo: Optional[PPO] = None):
        assert bool(path) ^ bool(ppo), "Either PPO model or path to PPO model must be provided"
        if path:
            ppo = PPO.load(path)
        self.beta = beta
        self.policy = ppo.policy
    
    def __call__(self, obs):
        features = self.policy.extract_features(obs)
        if self.policy.share_features_extractor:
            latent_pi, _ = self.policy.mlp_extractor(features)
        else:
            pi_features, _ = features
            latent_pi = self.policy.mlp_extractor.forward_actor(pi_features)
        logits = self.policy.action_net(latent_pi)
        
        # In most cases, mean_actions will be the logits before some kind of softmax.
        # We thus multiply them with 
    


