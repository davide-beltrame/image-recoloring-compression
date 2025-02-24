# Usage: glpsol --math k_means.mod --data k_means.dat

param n;     # Number of data points (unique colours)
param P;     # Number of possible clusters
param k;     # Number of clusters to select

param c{1..P};       # Cost of cluster j
param C{1..P, 1..n}, binary; # Cluster membership matrix

var y{1..P}, binary; # 1 if cluster j is selected

minimize TotalCost:
    sum {j in 1..P} c[j] * y[j];

s.t. CoverPoints {i in 1..n}:
    sum {j in 1..P} C[j,i] * y[j] = 1;

s.t. NumClusters:
    sum {j in 1..P} y[j] = k;

solve;

printf "\nTotal Cost: %g\n", TotalCost;

printf "Selected Clusters:\n";
for {j in 1..P: y[j] = 1} {
    printf "Cluster %d selected, Cost: %g\n", j, c[j];
    printf "Data points (unique colours): ";
    for {i in 1..n: C[j,i] = 1} {
        printf "%d ", i;
    }
    printf "\n\n";
}

end;