chessboard44 <- function(totalpoints = 4000, noisevariables = 6, noise = 0){
    chess44 <- array(0,c(totalpoints,noisevariables+3))

    for(i in 1:(noisevariables+2))
        chess44[,i] <- as.numeric(runif(dim(chess44)[1]))*4

    x <- chess44[,1]
    y <- chess44[,2]
    chess44[,ncol(chess44)] <- 0
    for(k in 1:4)  
       chess44[(x <= k & x > k-1 & y <= k & y > k-1), ncol(chess44)] <- 1
    for(k in 1:2)  
       chess44[(x <= k & x > k-1 & y <= k+2 & y > k+1), ncol(chess44)] <- 1
    for(k in 1:2)  
       chess44[(y <= k & y > k-1 & x <= k+2 & x > k+1), ncol(chess44)] <- 1

    if( noise > 0){
        flipclasslist <- sample(totalpoints, totalpoints * (noise / 100), replace = F)

        for(i in 1:length(flipclasslist)){
            if(chess44[flipclasslist[i], ncol(chess44)] == 1)
                chess44[flipclasslist[i], ncol(chess44)] = 0
            else if(chess44[flipclasslist[i], ncol(chess44)] == 0)
                chess44[flipclasslist[i], ncol(chess44)] = 1
        }
    }

    chess44 <- as.data.frame(chess44)
    chess44[,ncol(chess44)] <- as.factor(chess44[,ncol(chess44)])
    names(chess44)[9] <- "class"
    chess44
}



benchchessboard44 <- function(totalpoints = 4000, noisevariables = 6, noise = 0, nrealizations = 250, seed = 1000){
    set.seed(seed)
    accuracy_evtree <- array(-999999, nrealizations)
    accuracy_rpart <- array(-999999, nrealizations)
    accuracy_ctree <- array(-999999, nrealizations)

    NN_evtree <- array(-999999, nrealizations)
    NN_rpart   <- array(-999999, nrealizations)
    NN_ctree   <- array(-999999, nrealizations)

    print("realization no.:")
    for(f in 1:nrealizations){
        print(f)

        ch <- chessboard44(totalpoints = totalpoints, noisevariables = noisevariables, noise = noise )
        ch_train <- ch[1:(totalpoints/2), ]
        ch_test <-  ch[(totalpoints/2+1):totalpoints, ]
        xtrain <-   ch_train[, 1:(ncol(ch_train)-1)]
        ytrain <-   ch_train[, ncol(ch_train)]
        xpredict <- ch_test[, 1:(ncol(ch_test)-1)]
        ypredict <- ch_test[, ncol(ch_test)]

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
                        maxcompete = 0
                        )

        out_evtree <- evtree(ytrain ~ .,xtrain,
                        control = evtree.control(
                        minbucket = 7,
                        minsplit = 20,
                        maxdepth = 9,
                        alpha=1,
                        seed = seed
                        )
        )

        fit_ctree <- predict(out_ctree, newdata = xpredict)
        fit_evtree <- predict(out_evtree, newdata = xpredict)
        fit_rpart <- predict(out_rpart, newdata = xpredict, type = c("class"))

        accuracy_evtree[f] <- sum(fit_evtree == ypredict) / length(ypredict)
        accuracy_rpart[f] <- sum(fit_rpart == ypredict) / length(ypredict)
        accuracy_ctree[f] <- sum(fit_ctree == ypredict) / length(ypredict)

        NN_evtree[f] <- width(node_party(out_evtree))
        NN_rpart[f] <- length(out_rpart$splits[,4])+1
        NN_ctree[f] <- (max(where(out_ctree))+1)/2
    } # end for
    accuracy <- cbind(accuracy_evtree, accuracy_rpart, accuracy_ctree, NN_evtree, NN_rpart, NN_ctree)
    accuracy
}

