# Figure descriptions (text alternatives)

## convergence.png
Two line charts over 12 generations. Left: best fitness rises from 0.336 to 0.444 while hypervolume goes from 0.581 to 0.787. Right: mean mutation-rate gene goes from 0.350 to 0.326; the population multiplier ends at 2.50.

## fitness.png
Best, mean and worst fitness per generation with individual population members as grey dots. Best fitness ends at 0.444, mean at 0.388, worst at 0.211.

## accuracy_vs_runtime.png
Scatter of 181 evaluated architectures (validation accuracy vs training cost), coloured by generation discovered. A dashed line joins the 18 Pareto-optimal points. A star marks the selected model: 8 | relu | lr=0.03 with accuracy 0.788.

## pareto_front.png
Four panels showing the three-objective trade-off (accuracy, cost, parameters). Grey circles are dominated architectures, diamonds are the 18 Pareto-optimal ones, the star is the selected model with 195 parameters.
