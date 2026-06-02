| Feature                                   | K-Means++                                   | DBSCAN                               |
| ----------------------------------------- | ------------------------------------------- | ------------------------------------ |
| Type                                      | Centroid-based clustering                   | Density-based clustering             |
| Need to specify number of clusters?       | Yes, choose **K** beforehand                | No                                   |
| Cluster shape                             | Assumes roughly spherical/circular clusters | Can find arbitrary shapes            |
| Handles noise/outliers                    | Poorly                                      | Very well                            |
| Works well when cluster densities differ? | Usually no                                  | Can struggle if densities vary a lot |
| Main parameters                           | K (number of clusters)                      | ε (epsilon) and minPts               |
| Output                                    | Every point belongs to some cluster         | Some points may be labeled as noise  |


