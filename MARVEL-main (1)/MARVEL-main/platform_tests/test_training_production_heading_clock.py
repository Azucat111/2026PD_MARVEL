from pathlib import Path
import ast,types,textwrap,sys
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import train_extended_gppo_batched as trainer
from integrations.gppo.scheduler import GPPOTaskScheduler
from integrations.gppo.heading_features import HeadingObservations

def scheduler():
 s=GPPOTaskScheduler.__new__(GPPOTaskScheduler)
 s.runtime=types.SimpleNamespace(current_step=0)
 s.profile=types.SimpleNamespace(high_level_interval_steps=10,stale_age_fraction=.05)
 s.allocator=types.SimpleNamespace(adapter=types.SimpleNamespace())
 s._search_recovery=None
 return s

def actual_training_adapter(tmp_path):
 tree=ast.parse((ROOT/'scripts/train_extended_gppo_batched.py').read_text())
 cls=next(n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name=='TrainingAdapter')
 methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('setup','get_actions')]
 class Base:
  def setup(self):
   self.scheduler=scheduler();self.runtime=self.scheduler.runtime;return 'ready'
  def get_actions(self,observations):
   self.scheduler.observe(observations);return []
 namespace=dict(trainer.__dict__)
 namespace.update(original_adapter=Base,folder=tmp_path,append_motion_frame=lambda *a:None,capture_motion_frame=lambda *a:{})
 source='def factory():\n    episode_runtime=None\n    episode_scheduler=None\n    class TrainingAdapter(original_adapter):\n'+''.join(textwrap.indent(ast.unparse(n),'        ')+'\n' for n in methods)+'    return TrainingAdapter\n'
 exec(source,namespace);instance=namespace['factory']()();assert instance.setup()=='ready';return instance

def test_actual_training_control_tick_matches_production_staleness(tmp_path):
 adapter=actual_training_adapter(tmp_path);production=scheduler();observer=HeadingObservations(.05)
 production.allocator.adapter.observe=observer.observe
 for step in range(11):
  obs={7:{'heading':step*10.}};adapter.runtime.current_step=step;production.runtime.current_step=step
  adapter.get_actions(obs);production.observe(obs)
 assert observer.heading(7)==pytest.approx(95.)
 assert adapter.heading_observations.heading(7)==pytest.approx(observer.heading(7))

def test_duplicate_physics_observation_cannot_replace_control_history(tmp_path):
 adapter=actual_training_adapter(tmp_path)
 adapter.get_actions({7:{'heading':0.}})
 adapter.runtime.current_step=10;adapter.get_actions({7:{'heading':100.}})
 before=adapter.heading_observations.heading(7)
 adapter.get_actions({7:{'heading':110.}})
 assert adapter.heading_observations.heading(7)==pytest.approx(before)

def test_wraparound_uses_production_control_history(tmp_path):
 adapter=actual_training_adapter(tmp_path);adapter.get_actions({7:{'heading':350.}})
 for step in range(1,11):
  adapter.runtime.current_step=step;adapter.get_actions({7:{'heading':(350.+2*step)%360}})
 assert adapter.heading_observations.heading(7)==pytest.approx(9.)

def test_new_episode_does_not_reuse_heading_history(tmp_path):
 first=actual_training_adapter(tmp_path);first.get_actions({7:{'heading':10.}})
 first.runtime.current_step=10;first.get_actions({7:{'heading':100.}})
 second=actual_training_adapter(tmp_path);second.get_actions({7:{'heading':200.}})
 assert second.heading_observations.heading(7)==pytest.approx(200.)
 assert second.heading_observations is not first.heading_observations

def test_late_binding_gets_initial_snapshot_even_after_previous_observer(tmp_path):
 adapter=actual_training_adapter(tmp_path)
 adapter.scheduler._last_heading_observation_control_index=0
 adapter.heading_observations=trainer.bind_production_heading_observer(adapter.scheduler)
 adapter.get_actions({7:{'heading':17.}})
 assert adapter.heading_observations.heading(7)==pytest.approx(17.)
