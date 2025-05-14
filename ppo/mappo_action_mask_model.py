from gymnasium.spaces import Dict
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MIN
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.utils.torch_utils import one_hot as torch_one_hot
from ray.rllib.policy.view_requirement import ViewRequirement
from gymnasium.spaces import Box

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

class TorchActionMaskModelMappo(TorchModelV2, nn.Module):
    """PyTorch version of above TorchActionMaskModel."""

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        **kwargs,
    ):
        # Recover the original gym space before Rllib wrap
        orig_space = getattr(obs_space, "original_space", obs_space)

        print("ORIG SPACE", orig_space)

        
        # print("ORIG_SPACE:", orig_space["observations"])
        exit(0)

        

        assert (
            isinstance(orig_space, Dict)
            and "action_mask" in orig_space.spaces
            and "observations" in orig_space.spaces
        )

        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kwargs
        )
        nn.Module.__init__(self)

        # num_agents = 5
        # flat_obs_dim = obs_space["observations"].shape[0]
        # self.view_requirements["state"] = ViewRequirement(
        #     data_col="obs_flat",
        #     shift=0,
        #     batch_repeat_value=num_agents,
        #     space=Box(low=-float("inf"), high=float("inf"), shape=(num_agents * flat_obs_dim,))
        # )

        
        self.internal_model = TorchFC(
            orig_space["observations"],
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )

    
    def forward(self, input_dict, state, seq_lens):
        # print("INPUT DICT obs", input_dict["obs"], "len", len(input_dict["obs"]))
        # print("INPUT DICT obs flat", input_dict["obs_flat"])
        # print(input_dict["state"])
        
        '''
        action[b, a] == 1 -> action a is valid in batch_b
        action[b, a] == 0 -> action a is not valid
        '''
        action_mask = input_dict["obs"]["action_mask"]
        logits, _ = self.internal_model({"obs": input_dict["obs"]["observations"]})
        '''
        log(1) == 0 for valid actions
        log(0) == -inf for invalid actions
        torch.clamp() -> if -inf then take a very large neg. number
        '''
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        # For an invalid state perform logits - inf approx -inf
        masked_logits = logits + inf_mask
        return masked_logits, state

    def value_function(self):

        

        return self.internal_model.value_function()