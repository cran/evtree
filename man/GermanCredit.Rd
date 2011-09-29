\name{GermanCredit}
\alias{GermanCredit}
\title{Statlog German Credit}

\description{
The dataset contains data of past credit applicants. The applicants are rated
as \emph{good} or \emph{bad}. Models of this data can be used to determine if
new applicants present a \emph{good} or \emph{bad} credit risk. 
}

\usage{data("GermanCredit")}

\format{
A data frame containing 1,000 observations on 21 variables.
\describe{
  \item{status}{factor variable indicating the status of the existing checking account, with levels \code{... < 100 DM}, \code{0 <= ... < 200 DM}, \code{... >= 200 DM/salary for at least 1 year} and \code{no checking account}.}
  \item{duration}{duration in months.}  
   \item{credit_history}{factor variable indicating credit history, with levels \code{no credits taken/all credits paid back duly}, \code{all credits at this bank paid back duly},  \code{existing credits paid back duly till now}, \code{delay in paying off in the past} and \code{critical account/other credits existing}.}    
   \item{purpose}{factor variable indicating the credit's purpose, with levels \code{car (new)}, \code{car (used)}, \code{furniture/equipment}, \code{radio/television}, \code{domestic appliances}, \code{repairs}, \code{education},  \code{retraining}, \code{business} and \code{others}.}  
   \item{amount}{credit amount.}    
   \item{savings}{factor. savings account/bonds, with levels \code{... < 100 DM}, \code{100 <= ... < 500 DM}, \code{500 <= ... < 1000 DM}, \code{... >= 1000 DM} and \code{unknown/no savings account}.}  
   \item{employment_duration}{ordered factor indicating the duration of the current employment, with levels \code{unemployed}, \code{... < 1 year}, \code{1 <= ... < 4 years}, \code{4 <= ... < 7 years} and \code{... >= 7 years}.}    
   \item{installment_rate}{installment rate in percentage of disposable income.}  
   \item{personal_status_sex}{factor variable indicating personal status and sex, with levels 
   \code{male:divorced/separated}, \code{female:divorced/separated/married}, 
   \code{male:single}, \code{male:married/widowed} and \code{female:single}.}    
   \item{other_debtors}{factor. Other debtors, with levels \code{none}, \code{co-applicant} and \code{guarantor}.}      
   \item{present_residence}{present residence since?}    
   \item{property}{factor variable indicating the client's highest valued property, with levels \code{real estate}, \code{building society savings agreement/life insurance}, \code{car or other} and \code{unknown/no property}.}  
   \item{age}{client's age.}    
   \item{other_installment_plans}{factor variable indicating other installment plans, with levels \code{bank}, \code{stores} and \code{none}.}     
   \item{housing}{factor variable indicating housing, with levels \code{rent}, \code{own} and \code{for free}.}    
   \item{number_credits}{number of existing credits at this bank.}  
   \item{job}{factor indicating employment status, with levels \code{unemployed/unskilled - non-resident}, \code{unskilled - resident}, \code{skilled employee/official} and \code{management/self-employed/highly qualified employee/officer}.}    
   \item{people_liable}{Number of people being liable to provide maintenance.}     
   \item{telephone}{binary variable indicating if the customer has a registered telephone number.}
   \item{foreign_worker}{binary variable indicating if the customer is a foreign worker.}
   \item{credit_risk}{binary variable indicating credit risk, with levels \code{good} and \code{bad}.}
}
}

\source{
The original data was provided by:

Professor Dr. Hans Hofmann,  
Institut fuer Statistik und Oekonometrie,
Universitaet Hamburg,  
FB Wirtschaftswissenschaften,  
Von-Melle-Park 5,    
2000 Hamburg 13
   
The dataset has been taken from the UCI Repository Of Machine Learning Databases at

\url{http://archive.ics.uci.edu/ml/}.
}

\details{
The use of a cost matrix is suggested for this dataset. It is worse to class a customer as good when they are bad (cost = 5), than it is to class a customer as bad when they are good (cost = 1). 
}

\examples{
data("GermanCredit")
summary(GermanCredit)
\dontrun{
gcw <- array(1, nrow(GermanCredit))
gcw[GermanCredit$credit_risk == "bad"] <- 5
gct <- evtree(credit_risk ~ . , data = GermanCredit, weights = gcw) 
gct
table(predict(gct), GermanCredit$credit_risk)
plot(gct)
}
}

\keyword{datasets}
