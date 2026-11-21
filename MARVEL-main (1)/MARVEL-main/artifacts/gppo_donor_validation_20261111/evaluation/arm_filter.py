"""Arm separation: the frozen donor filter is called only by donor."""
from donor_filter import filter_search_donors

def graph_for_arm(graph,arm,comm_range=20.):
 if arm not in ('rule','unfiltered','donor'):raise ValueError('Unknown frozen arm')
 if arm!='donor':return graph,None
 return filter_search_donors(graph,comm_range)
