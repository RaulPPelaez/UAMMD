#!/usr/bin/Rscript
#---------------------------------------------------------#
#                         density.R                       #
#---------------------------------------------------------#

# Density profiles for a sequence of frames with coordinate values.

# Calculation parameters
filename <- "rand.dat"; # Input data file
outfile <- "rand.out"; # Output data file
lineskip <- 0; # Number of comment lines at the beginning of the data file.
np <- 100000; # Number of particles per frame
minval <- -67; # Lower limit of the histogram
maxval <- 67; # Upper limit of the histogram
binsize <- 4.0; # Histogram bin size

# Initialise program parameters
# Read in the values from a file.
data <- as.matrix(read.table(filename, skip = lineskip));

# Set output file
sink(outfile)

# Get the number of rows, columns and frames
n <- nrow(data);

# Create matrices to store histogram
histrows <- ceiling((maxval - minval)/binsize);
histcols <- ceiling((maxval - minval)/binsize);
histogram <- matrix(rep(0, histrows*histcols), histrows);

# Output header
cat("# ---- Histogram ---- #\n#\n");

# Read data and store in histogram.
for(i in (1:n))
{

# Find the bin for the data point.
binx <- ceiling((data[i,1] - minval)/binsize);
biny <- ceiling((data[i,2] - minval)/binsize);

# If the point is between minval and maxval, add it to the histogram.
if((binx > 0) & (binx <= histrows))
  if((biny > 0) & (biny <= histrows))
    histogram[binx, biny] <- histogram[binx, biny] + 1;

}



# Output the histogram
for(i in (1:histrows))
{
  for(j in (1:histcols)) {
    cat(minval+binsize*(i-0.5), "\t", minval+binsize*(j-0.5), "\t", histogram[i,j]/(np*histrows*histcols*binsize*binsize), "\n");
  }

  cat("\n");
}

