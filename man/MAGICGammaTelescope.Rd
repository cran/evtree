\name{MAGICGammaTelescope}
\alias{MAGICGammaTelescope}
\title{MAGIC Gamma Telescope}

\description{
The data was generated to simulate registration of high energy gamma particles in a Major Atmospheric Gamma-Ray Imaging Cherenkov (MAGIC) Gamma Telescope. The task is to distinguish \emph{gamma rays} (signal) from \emph{hadronic showers} (background).
}

\usage{data("MAGICGammaTelescope")}

\format{
A data frame containing 19,020 observations on 11 variables.
\describe{
  \item{fLength}{major axis of ellipse [mm].}
  \item{fWidth}{minor axis of ellipse [mm].}
  \item{fSize}{10-log of sum of content of all pixels [in #phot].}
  \item{fConc}{ratio of sum of two highest pixels over fSize [ratio].}
  \item{fConc1}{ratio of highest pixel over fSize [ratio].}
  \item{fAsym}{distance from highest pixel to center, projected onto major axis [mm].}
  \item{fM3Long}{3rd root of third moment along major axis [mm].}
  \item{fM3Trans}{3rd root of third moment along minor axis [mm].}
  \item{fAlpha}{angle of major axis with vector to origin [deg].}
  \item{fDist}{distance from origin to center of ellipse [mm].}
  \item{class}{binary variable class, with levels \code{gamma} (signal) and \code{hadron} (background).}
}
}

\source{
	The original data was provided by:
	
	
	R. K. Bock,
	Major Atmospheric Gamma Imaging Cherenkov Telescope project (MAGIC),        
	rkb '@' mail.cern.ch,
	\url{http://www.magic.mppmu.mpg.de}

	and was donated by:

	P. Savicky,	
	Institute of Computer Science, AS of CR, Czech Republic,
	savicky '@' cs.cas.cz 


	The dataset has been taken from the UCI Repository Of Machine Learning Databases at


	\url{http://archive.ics.uci.edu/ml/}.
}

\details{
Classifying a background event as signal is worse than classifying a signal event as background. For a meaningful comparison of different classifiers the use of an ROC curve with thresholds 0.01, 0.02, 0.05, 0.1, 0.2 is suggested.
}


\references{
Bock, R.K., Chilingarian, A., Gaug, M., Hakl, F., Hengstebeck, T., Jirina, M., Klaschka, J., Kotrc, E., Savicky, P., Towers, S., Vaicilius, A., Wittek W. (2004). Methods for Multidimensional event Classification: a Case Study Using Images From a Cherenkov Gamma-Ray Telescope. \emph{Nuclear Instruments and Methods in Physics Research Section A: Accelerators, Spectrometers, Detectors and Associated Equipment}, \emph{516}(1), 511--528.

P. Savicky, E. Kotrc (2004). Experimental Study of Leaf Confidences for Random Forest. In \emph{Proceedings of COMPSTAT}, pp. 1767--1774. Physica Verlag, Heidelberg, Germany. 

J. Dvorak, P. Savicky (2007). Softening Splits in Decision Trees Using Simulated Annealing. In \emph{Proceedings of the 8th International Conference on Adaptive and Natural Computing Algorithms, Part I}, pp. 721--729, Springer-Verlag, New-York.
}

\examples{
data("MAGICGammaTelescope")
summary(MAGICGammaTelescope)
\dontrun{
set.seed(1090)
mgtt <- evtree(class ~ . , data = MAGICGammaTelescope) 
mgtt
table(predict(mgtt), MAGICGammaTelescope$class)
plot(mgtt)
}
}

\keyword{datasets}
