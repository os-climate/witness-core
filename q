[1mdiff --git a/climateeconomics/tests/l0_test_dashboard.py b/climateeconomics/tests/l0_test_dashboard.py[m
[1mindex b8f9a198..37db9ad3 100644[m
[1m--- a/climateeconomics/tests/l0_test_dashboard.py[m
[1m+++ b/climateeconomics/tests/l0_test_dashboard.py[m
[36m@@ -71,7 +71,7 @@[m [mclass PostProcessEnergy(unittest.TestCase):[m
                                                               as_json=False)[m
 [m
             for graph in graph_list:[m
[31m-                #graph.to_plotly().show()[m
[32m+[m[32m                graph.to_plotly().show()[m
                 pass[m
 [m
 [m
