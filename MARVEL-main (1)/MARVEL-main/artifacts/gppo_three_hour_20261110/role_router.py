"""Experimental frozen policy composition; no model blending or training."""
from integrations.gppo.task_graph import TaskType

def role_key(graph):
 roles={t.task_type for t in graph.subtasks}
 if not roles:return 'round8'
 if roles=={TaskType.TARGET_SEARCH}:return 'round17'
 if roles=={TaskType.RELAY}:return 'round8'
 raise ValueError('This frozen composition supports pure Search/Relay catalogs only')

def assign_with_model(adapter,original,model,graph,deterministic):
 previous=adapter.model
 try:
  adapter.model=model
  return original(graph,deterministic)
 finally:adapter.model=previous

def install(adapter,search_model):
 original=adapter.assign;models={'round8':adapter.model,'round17':search_model}
 def assign(graph,deterministic=True):
  key=role_key(graph);adapter.last_role_policy=key
  return assign_with_model(adapter,original,models[key],graph,deterministic)
 adapter.assign=assign
