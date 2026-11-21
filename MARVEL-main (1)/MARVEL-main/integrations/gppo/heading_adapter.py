"""Explicit extended heading protocol; frozen loader and profile remain independent."""
from pathlib import Path
import hashlib
import torch
from .adapter import GPPOInferenceAdapter,GPPOAssignment
from .gppo_model import GPPOActorCritic
from .heading_features import HEADING_PROTOCOL,HeadingObservations,heading_batch
from .opportunity_features import OPPORTUNITY_PROTOCOL,opportunity_batch

class ExtendedGPPOInferenceAdapter(GPPOInferenceAdapter):
    def __init__(self, base_checkpoint_path, candidate_checkpoint_path, device='cpu', *, expected_sha256=None):
        path=Path(candidate_checkpoint_path).expanduser().resolve()
        sha=hashlib.sha256(path.read_bytes()).hexdigest()
        if expected_sha256 is not None and sha!=str(expected_sha256):raise ValueError('Experimental candidate hash mismatch')
        candidate=torch.load(path,map_location=device,weights_only=False)
        if not isinstance(candidate,dict) or candidate.get('protocol_version') not in (HEADING_PROTOCOL,OPPORTUNITY_PROTOCOL):
            raise ValueError('Explicit supported extended candidate protocol required')
        super().__init__(base_checkpoint_path,device=device)
        self.model=GPPOActorCritic(edge_feature_dim=6 if candidate['protocol_version']==HEADING_PROTOCOL else 9).to(self.device)
        self.model.load_state_dict(candidate['model'],strict=True);self.model.eval()
        self.model_protocol=candidate['protocol_version'];self.candidate_sha256=sha
        self.candidate_checkpoint_path=str(path)
        mixed=self.checkpoint['mixed_config']
        self.heading_observations=HeadingObservations(float(mixed['comm_delay_ms'])/float(mixed['control_step_ms']))
        self.last_batch=None

    def observe(self, observations):
        self.heading_observations.observe(observations)

    @torch.no_grad()
    def assign(self, graph, deterministic=True):
        if not graph.num_uavs or not graph.num_subtasks:return None
        if self.model_protocol==OPPORTUNITY_PROTOCOL:
            provider=getattr(self,'public_search_context_provider',None)
            if provider is None:raise RuntimeError('Opportunity candidate requires public scheduler context')
            self.heading_observations.public_search_context=provider()
            batch=opportunity_batch(graph,self.heading_observations,self.device)
        else:
            batch=heading_batch(graph,self.heading_observations,self.device)
        self.last_batch={key:value.detach().clone() for key,value in batch.items()}
        output=self.model(**batch)
        if not bool(output['has_valid_action'][0]):return None
        action=(int(output['flat_logits'][0].argmax()) if deterministic else
            int(torch.distributions.Categorical(logits=output['flat_logits'][0]).sample()))
        if bool(batch['action_mask'].reshape(-1)[action]):raise RuntimeError('Candidate selected masked action')
        ti,ui=divmod(action,graph.num_uavs)
        return GPPOAssignment(action,ti,ui,int(graph.subtasks[ti].subtask_id),int(graph.uav_states[ui].uav_id),float(output['value'][0]))
