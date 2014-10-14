\name{BBBClub}
\alias{BBBClub}

\title{Bookbinder's Book Club}

\description{
  Marketing case study about a (fictitious) American book club to whose
  customers a book about \dQuote{The Art History of Florence} was advertised.
}

\usage{data("BBBClub")}

\format{
A data frame containing 1,300 observations on 11 variables.
\describe{
  \item{choice}{factor. Did the customer buy the advertised book?}
  \item{gender}{factor indicating gender.}  
  \item{amount}{total amount of money spent at the BBB Club.}  
  \item{freq}{number of books purchased at the BBB Club.}  
  \item{last}{number of months since the last purchase.}  
  \item{first}{number of months since the first purchase.}  
  \item{child}{number of children's books purchased.}  
  \item{youth}{number of youth books purchased.}  
  \item{cook}{number of cookbooks purchased.}  
  \item{diy}{number of do-it-yourself books purchased.}  
  \item{art}{number of art books purchased.}  
}
}

\details{
  The data is a marketing case study about a (fictitious) American book club. 
  taken from the \emph{Marketing Engineering} textbook of Lilien and Rangaswamy (2004).  
  In this case study, a brochure of the book \dQuote{The Art History of Florence} was
  sent to 20,000 customers and 1,806 of which bought the book. A subsample of 1,300
  customers is provided in \code{BBBClub} for building a predictive model for \code{choice}.

  The use of a cost matrix is suggested for this dataset.	
  Classifying a customer that purchased the book as a non-buyer is worse (cost = 5),
  than it is to classify a custumer that did not purchase the book as a buyer (cost = 1). 
}

\source{
  Complements to Lilien and Rangaswamy (2004).
}

\references{
  Lilien GL, Rangaswamy A (2004).
    \emph{Marketing Engineering: Computer-Assisted Marketing Analysis and Planning},
    2nd edition. Victoria, BC: Trafford Publishing. 
}

\examples{
\dontrun{
## data, packages, random seed
data("BBBClub", package = "evtree")
library("rpart")
set.seed(1090)

## learn trees
ev <- evtree(choice ~ ., data = BBBClub, minbucket = 10, maxdepth = 2)
rp <- as.party(rpart(choice ~ ., data = BBBClub, minbucket = 10))
ct <- ctree(choice ~ ., data = BBBClub, minbucket = 10, mincrit = 0.99)

## visualization
plot(ev)
plot(rp)
plot(ct)

## accuracy: misclassification rate
mc <- function(obj) 1 - mean(predict(obj) == BBBClub$choice)
c("evtree" = mc(ev), "rpart" = mc(rp), "ctree" = mc(ct))

## complexity: number of terminal nodes
c("evtree" = width(ev), "rpart" = width(rp), "ctree" = width(ct))

## compare structure of predictions
ftable(tab <- table(evtree = predict(ev), rpart  = predict(rp),
  ctree  = predict(ct), observed = BBBClub$choice))

## compare customer predictions only (absolute, proportion correct)
sapply(c("evtree", "rpart", "ctree"), function(nam) {
  mt <- margin.table(tab, c(match(nam, names(dimnames(tab))), 4))
  c(abs = as.vector(rowSums(mt))[2],
    rel = round(100 * prop.table(mt, 1)[2, 2], digits = 3))
})
}
}
\keyword{datasets}
