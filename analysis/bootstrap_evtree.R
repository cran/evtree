bootstrap_evtree <- function(kdata, nboots = 250, seed = 1000){
    set.seed(seed)
    accuracy_evtree <- array(-999999, nboots)
    accuracy_crpart <- array(-999999, nboots)
    accuracy_ctree <- array(-999999, nboots)

    NN_evtree <- array(-999999, nboots)
    NN_rpart   <- array(-999999, nboots)
    NN_ctree   <- array(-999999, nboots)

    print("bootstrap no.:")
    for(f in 1:nboots){
        print(f)
        rand <- sample(nrow(kdata), replace = T)
        rand <- sort(rand)
        nbootTraining <- kdata[rand, ]
        s <- 1
        k <- 1
        flag <- FALSE
        while(k <= dim(kdata)[1]){
            if(rand[k] == s ){
                s <- s + 1
                k <- k + 1
            }else if(rand[k] < s ){
                k <- k + 1
            }else if(rand[k] > s ){
                if(flag == FALSE){
                  nbootTest <- kdata[s,]
                  flag <- TRUE
                }else{
                  nbootTest <- rbind(nbootTest, kdata[s,])
                }
                s <- s + 1
            }
        }

        xtrain <- nbootTraining[ ,1:(dim(nbootTraining)[2]-1)]
        ytrain <- nbootTraining[ ,dim(nbootTraining)[2]]
        xpredict <- nbootTest[ ,1:(dim(nbootTest)[2]-1)]
        ypredict <- nbootTest[ ,dim(nbootTest)[2]]
        out_ctree <- party::ctree(
                ytrain ~ ., data = cbind(xtrain, ytrain),
                control = party::ctree_control(
                       minbucket = 7,
                       minsplit = 20,
                       maxdepth = 9
                )
        )

        out_rpart <- rpart(
                ytrain ~ ., data = xtrain,
                        minbucket = 7,
                        minsplit = 20,
                        maxdepth = 9,
                        maxsurrogate = 0,
                        maxcompete = 0,
                        )

        out_evtree <- evtree(ytrain ~ .,xtrain,
                        control = evtree.control(
                        minbucket = 7,
                        minsplit = 20,
                        maxdepth = 9,
                        seed = seed
                        )
        )

        fit_ctree <- predict(out_ctree, newdata= xpredict)
        fit_evtree <- predict(out_evtree, newdata= xpredict)
        if(is.factor(ypredict)){
            fit_rpart <- predict(out_rpart, newdata= xpredict, type = c("class"))
            accuracy_evtree[f] <- sum(fit_evtree == ypredict) / length(ypredict)
            accuracy_crpart[f] <- sum(fit_rpart == ypredict) / length(ypredict)
            accuracy_ctree[f] <- sum(fit_ctree == ypredict) / length(ypredict)
        }else{
            fit_rpart <- predict(out_rpart, newdata= xpredict)
            accuracy_evtree[f] <- sum((fit_evtree - ypredict)^2) / length(ypredict)
            accuracy_crpart[f] <- sum((fit_rpart - ypredict)^2) / length(ypredict)
            accuracy_ctree[f] <- sum((fit_ctree - ypredict)^2) / length(ypredict)
        }

        NN_evtree[f] <- width(node_party(out_evtree))
        NN_rpart[f] <- length(out_rpart$splits[,4])+1
        NN_ctree[f] <- (max(where(out_ctree))+1)/2
        
        
 	if(sum(predict(out_evtree, newdata = unclass(out_evtree)$data) == ytrain) != sum(predict(out_evtree) == ytrain))
 		print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Failure in prediction of cats")
        
    } # end for
    accuracy <- cbind(accuracy_evtree, accuracy_crpart, accuracy_ctree, NN_evtree, NN_rpart, NN_ctree)
    accuracy
}
