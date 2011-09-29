\name{StatlogHeart}
\alias{StatlogHeart}
\title{Statlog Heart}

\description{
Models of this data predict the \emph{absence} or \emph{presence} of heart disease.
}

\usage{data("StatlogHeart")}

\format{
A data frame containing 270 observations on 14 variables.
\describe{
  \item{age}{age in years.}
  \item{sex}{binary variable indicating sex.}  
  \item{chest_pain_type}{factor variable indicating the chest pain type, with levels \code{typical angina}, \code{atypical angina}, \code{non-anginal pain} and \code{asymptomatic}.}  
  \item{resting_blood_pressure}{resting blood pressure.}    
  \item{serum_colestoral}{serum cholesterol in mg/dl. }  
  \item{fasting_blood_sugar}{binary variable indicating if fasting blood sugar > 120 mg/dl.}     
  \item{resting_electrocardiographic_results}{factor variable indicating resting electrocardiographic results, with levels \code{0}: normal, \code{1}: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV) and \code{2}: showing probable or definite left ventricular hypertrophy by Estes' criteria.}  
  \item{maximum_heart_rate}{the maximum heart rate achieved.}    
  \item{exercise_induced_angina}{binary variable indicating the presence of exercise induced angina.}  
  \item{oldpeak}{oldpeak = ST depression induced by exercise relative to rest.}      
  \item{slope_of_the_peak}{ordered factor variable describing the slope of the peak exercise ST segment, with levels \code{upsloping}, \code{flat} and \code{downsloping}.}  
  \item{major_vessels}{number of major vessels colored by flouroscopy.}    
  \item{thal}{factor variable thal, with levels \code{normal}, \code{fixed defect} and \code{reversible defect}.}  
  \item{heart_disease}{binary variable indicating the \code{presence} or \code{absence} of heart disease.}      
}
}

\source{
The dataset has been taken from the UCI Repository Of Machine Learning Databases at

\url{http://archive.ics.uci.edu/ml/}.
}

\details{
The use of a cost matrix is suggested for this dataset. It is worse to class patients with heart disease as patients without heart disease (cost = 5), than it is to class patients without heart disease as having heart disease (cost = 1). 
}

\examples{
data("StatlogHeart")
summary(StatlogHeart)
shw <- array(1, nrow(StatlogHeart))
shw[StatlogHeart$heart_disease == "presence"] <- 5
sht <- evtree(heart_disease ~ . , data = StatlogHeart, weights = shw) 
sht
table(predict(sht), StatlogHeart$heart_disease)
plot(sht)
}

\keyword{datasets}
