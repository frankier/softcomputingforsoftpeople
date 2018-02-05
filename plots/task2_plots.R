#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

data = read.csv(args[1])
png(args[2])
colors <- c("black", "red", "blue")[data$type + 1]
plot(data[c("x1", "x2")], xlim=c(-1, 2), ylim=c(-1, 1), col=colors, pch=20, ann=FALSE)
