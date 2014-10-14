\name{evtree}
\alias{evtree}
\alias{evtree-package}

\title{
Evolutionary Learning of Globally Optimal Trees
}

\description{
Learning of globally optimal classification and regression trees by
using evolutionary algorithms.	
}

\usage{
evtree(formula, data, subset, na.action, weights,
  control = evtree.control(\dots), \dots)
}

\arguments{
  \item{formula}{a symbolic description of the model to be fit,
    no interactions should be used.}
  \item{data, subset, na.action}{arguments controlling formula processing
    via \code{\link[stats]{model.frame}}.}
  \item{weights}{optional integer vector of case weights.}
  \item{control}{a list of control arguments specified via
    \code{\link{evtree.control}}.}
  \item{\dots}{arguments passed to \code{\link{evtree.control}}.}
}

\details{
Globally optimal classification and regression trees are learned by using
evolutionary algorithm. Roughly, the algorithm works as follows. First, a set of
trees is initialized with random split rules in the root nodes. Second, mutation
and crossover operators are applied to modify the trees' structure and the tests
that are applied in the internal nodes. After each modification step a survivor
selection mechanism selects the best candidate models for the next iteration. In
this evolutionary process the mean quality of the population increases over
time. The algorithm terminates when the quality of the best trees does not
improve further, but not later than a maximum number of iterations specified by
\code{niterations} in \code{\link{evtree.control}}.

More details on the algorithm are provided Grubinger et al. (2014) which is also
provided as \code{vignette("evtree", package = "evtree")}. 

The resulting trees can be summarized and visualized by the \code{\link[partykit]{print.constparty}},
and \code{\link[partykit]{plot.constparty}} methods provided by the \pkg{partykit} package.
Moreover, the \code{\link[partykit]{predict.party}} method can be used to compute fitted responses,
probabilities (for classification trees), and nodes.
}

\value{
  An object of class \code{\link[partykit]{party}}.
}

\references{
Grubinger T, Zeileis A, Pfeiffer KP (2014).
  evtree: Evolutionary Learning of Globally Optimal Classification and Regression Trees in R.
  \emph{Journal of Statistical Software}, \bold{61}(1), 1-29.
  \url{http://www.jstatsoft.org/v61/i01/}
}

\examples{
## regression
set.seed(1090)
airq <- subset(airquality, !is.na(Ozone) & complete.cases(airquality))
ev_air <- evtree(Ozone ~ ., data = airq)
ev_air
plot(ev_air)
mean((airq$Ozone - predict(ev_air))^2)

## classification
ev_iris <- evtree(Species ~ .,data = iris)
ev_iris
plot(ev_iris)
table(predict(ev_iris), iris$Species)
1 - mean(predict(ev_iris) == iris$Species)
}

\keyword{tree}
