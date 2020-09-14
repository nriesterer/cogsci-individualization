New Analysis
============

This reconducted analysis is based on an updated version of the pymreasoner wrapper. Instead of querying mReasoner directly for predictions (which resulted in infeasible runtimes on some systems), it now uses a cached variant and improved parameter handling.

Most Notable Changes
--------------------

1. mReasoner's coverage accuracies rise, especially in the case of individualized fitting where it now exceeds the performance of PHM
2. Epsilon and lambda are less irrelevant now. Instead omega shows the most uniform distributions of the parameters suggesting lacking importance of the weakening mechanism. This, in turn, can be explained by the dominance of system 1 configurations (low values for the sigma parameter), in which cases omega has no effect.
