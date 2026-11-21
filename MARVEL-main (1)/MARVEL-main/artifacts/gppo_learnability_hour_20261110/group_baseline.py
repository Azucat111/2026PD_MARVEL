"""Training-only leave-one-out episode baseline; no production import."""
import math
def episode_advantages(completions):
 c=list(map(float,completions))
 if len(c)<2 or any(not math.isfinite(v) or not 0<=v<=300 for v in c):raise ValueError('Need at least2 fullH300 outcomes')
 # G_i(t)=-(C_i-t)/300, b_i(t)=-(mean_other_C-t)/300;
 # difference divided by fixed0.05 cancels t. No fitted state critic.
 return [(sum(c[:i]+c[i+1:])/(len(c)-1)-v)/15. for i,v in enumerate(c)]
