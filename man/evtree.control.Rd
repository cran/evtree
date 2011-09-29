\name{evtree.control}
\alias{evtree.control}
\title{Control for evtree}
\description{
  Various parameters that control aspects of the \code{evtree} fit.
}
\usage{
evtree.control(minbucket = 7L, minsplit = 20L, maxdepth = 9L, 
  niterations = 10000L, ntrees = 100L, alpha = 1,
  operatorprob = list(pmutatemajor = 0.2, pmutateminor = 0.2,
    pcrossover = 0.2, psplit = 0.2, pprune = 0.2),
  seed = NULL, \dots)
}

\arguments{
  \item{minbucket}{the minimum sum of weights in a terminal node. }
  \item{minsplit}{the minimum sum of weights in a node in order to
    be considered for splitting. }
  \item{maxdepth}{maximum depth of the tree. Note, that the memory
    requirements increase by the square of the m'aximum tree depth.}
  \item{niterations}{in case the run does not converge, it terminates
    after a specified number of iterations defined by \code{niterations}.}
  \item{ntrees}{the number of trees in the population.}
  \item{alpha}{regulates the complexity part of the cost function.
    Increasing values of alpha encourage decreasing tree sizes.}
  \item{operatorprob}{list or vector of probabilities for the selection
    of variation operators. May also be specified partially in which case
    the default values are still used for the unspecified arguments.
    Always scaled to sum to 100 percent.} 
  \item{seed}{an numeric seed to initialize the random number generator
    (for reproducibility). By default the seed is randomly drawn using
    \code{\link[stats]{runif}} in order to inherit the state of
    \code{\link[base]{.Random.seed}}. If set to \code{seed = -1L},    
    the random number generator is initialized by the system time.}
  \item{\dots}{additional arguments. }
}

\value{
  A list with the (potentially processed) control parameters.
}

\keyword{misc}
