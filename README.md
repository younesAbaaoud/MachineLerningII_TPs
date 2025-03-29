# Machine Learning II – Travaux Pratiques en Reinforcement Learning


## Contexte

Ce projet consiste à appliquer des algorithmes de **Reinforcement Learning** tels que **Q-learning**, **SARSA** et **PPO** dans divers environnements de simulation. L'objectif est de développer des agents capables d'apprendre des stratégies optimales dans des environnements tels que **CartPole-v1**, **Taxi-v3**, et un environnement personnalisé de gestion de trafic. Ce TP vise à illustrer les principes de base du **Reinforcement Learning** et de la **mise en œuvre de politiques d'apprentissage** à travers des simulations.



## Objectifs du TP

- Implémenter et tester des algorithmes de **Reinforcement Learning** tels que **Q-learning**, **SARSA**, et **PPO**.
- Appliquer ces algorithmes dans plusieurs environnements, dont **CartPole-v1**, **Taxi-v3**, et un environnement personnalisé de gestion de trafic.
- Analyser les performances des agents et visualiser les résultats via des graphiques.

## Résultats Obtenus

- **Q-Learning sur une grille :** L'agent apprend à se déplacer dans une grille pour atteindre un trésor tout en évitant des pièges. L'algorithme montre une convergence stable vers la politique optimale après quelques milliers d'épisodes.
  
- **CartPole-v1 :** L'agent utilise Q-Learning et SARSA pour apprendre à équilibrer une barre en utilisant une méthode d'exploration-exploitation. Les résultats montrent que **SARSA** a une convergence plus douce que **Q-Learning**, mais les deux approches réussissent à stabiliser la barre dans la plupart des essais.

- **Traffic Environment :** L'agent apprend à gérer le trafic dans un environnement simulé, en maximisant les récompenses liées à l'efficacité du trafic. Cette partie montre comment les agents peuvent être utilisés dans des contextes plus complexes et personnalisés.

- **Taxi-v3 avec PPO :** L'algorithme **PPO** a été utilisé pour enseigner à l'agent à transporter des passagers dans un environnement de taxi. L'agent améliore progressivement ses performances, atteignant une politique stable après plusieurs milliers d'interactions.

## Remarques Techniques

1. **Environnement CartPole :**
   - L'implémentation de l'agent CartPole utilise la bibliothèque **gymnasium**, qui permet de facilement configurer et tester des agents dans des environnements standard.
   - Le rendu graphique peut être activé pour observer l'évolution de l'agent pendant l'exécution en définissant le mode `render_mode="human"` dans la fonction `gym.make()`.

2. **Algorithmes de Q-Learning et SARSA :**
   - Ces algorithmes utilisent des **tables Q** pour estimer les valeurs d'action dans chaque état. Un taux d'apprentissage et un facteur de discount sont utilisés pour ajuster ces valeurs au fur et à mesure des interactions avec l'environnement.

3. **PPO sur Taxi-v3 :**
   - **PPO** est un algorithme plus complexe qui nécessite l'utilisation de politiques stochastiques. Il est particulièrement adapté aux environnements où l'agent doit explorer de grandes espaces d'actions.
   
4. **Environnement de Trafic :**
   - Un environnement personnalisé a été utilisé pour simuler un réseau de trafic. L'agent doit apprendre à naviguer efficacement en optimisant le trafic et les déplacements.



## Conclusion

Ce projet illustre l'application des algorithmes de **Reinforcement Learning** dans des environnements variés, allant de la gestion de grille simple avec **Q-Learning** à des environnements plus complexes comme **CartPole-v1**, **Taxi-v3**, et un environnement personnalisé de gestion du trafic. Les résultats obtenus montrent que ces algorithmes sont capables d'apprendre des stratégies optimales, mais que chaque environnement présente des défis uniques pour l'agent.

- **Q-Learning** et **SARSA** sont efficaces pour des problèmes simples comme la grille et CartPole, mais peuvent rencontrer des difficultés dans des environnements plus dynamiques.
- **PPO**, qui est plus adapté à des environnements complexes avec des actions continues et stochastiques, a montré de bonnes performances sur Taxi-v3.

L'ensemble du projet démontre bien les capacités du **Reinforcement Learning** à résoudre des problèmes complexes, et souligne l'importance de choisir l'algorithme approprié en fonction de l'environnement et de la tâche. Les visualisations des performances des agents montrent l'évolution de leur apprentissage et leur capacité à trouver des solutions optimales.

En conclusion, ce TP a permis de mieux comprendre les principes fondamentaux du Reinforcement Learning et d'explorer différentes approches pour résoudre des problèmes complexes en intelligence artificielle.
