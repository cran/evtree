\name{ContraceptiveChoice}
\alias{ContraceptiveChoice}
\title{Contraceptive Method Choice}

\description{
Data of married women who were either not pregnant or do not know if they were
at the time of interview. The task is to predict the women's current
contraceptive method choice (\emph{no use}, \emph{long-term methods},
\emph{short-term methods}) based on her demographic and socio-economic
characteristics.
}

\usage{data("ContraceptiveChoice")}

\format{
A data frame containing 1,437 observations on 10 variables.
\describe{
  \item{wifes_age}{wife's age in years.}
  \item{wifes_education}{ordered factor indicating the wife's education, with levels \code{"low"}, \code{"medium-low"}, \code{"medium-high"} and \code{"high"}.}  
  \item{husbands_education}{ordered factor indicating the wife's education, with levels \code{"low"}, \code{"medium-low"}, \code{"medium-high"} and \code{"high"}.}
  \item{number_of_children}{number of children.}  
  \item{wifes_religion}{binary variable indicating the wife's religion, with levels \code{"non-Islam"} and \code{"Islam"}.}
  \item{wife_now_working}{binary variable indicating if the wife is working.}    
  \item{husbands_occupation}{ordered factor indicating the husbands occupation, with levels \code{"low"}, \code{"medium-low"}, \code{"medium-high"} and \code{"high"}. }  
  \item{standard_of_living_index}{standard of living index with levels \code{"low"}, \code{"medium-low"}, \code{"medium-high"} and \code{"high"}.}  
  \item{media_exposure}{binary variable indicating media exposure, with levels \code{"good"} and \code{"not good"}.}
  \item{contraceptive_method_used}{factor variable indicating the contraceptive method used, with levels \code{"no-use"}, \code{"long-term"} and \code{"short-term"}.}
}
}

\source{
This dataset is a subset of the 1987 National Indonesia Contraceptive Prevalence Survey and was created by Tjen-Sien Lim.
    
It has been taken from the UCI Repository Of Machine Learning Databases at

\url{http://archive.ics.uci.edu/ml/}.
}

\references{
Lim, T.-S., Loh, W.-Y. & Shih, Y.-S. (1999). A Comparison of Prediction Accuracy, Complexity, and Training Time of Thirty-three Old and New Classification Algorithms. \emph{Machine Learning}, \bold{40}(3), 203--228. 
}

\examples{
data("ContraceptiveChoice")
summary(ContraceptiveChoice)
\dontrun{
set.seed(1090)
contt <- evtree(contraceptive_method_used ~ . , data = ContraceptiveChoice) 
contt
table(predict(contt), ContraceptiveChoice$contraceptive_method_used)
plot(contt)
}
}

\keyword{datasets}
