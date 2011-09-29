#include "container.h"

static Container* container;

extern "C"{
void tree(int* nInst, int* nVar, int *varType, double* ndata, int* weights, int* prediction, int *splitV, double *splitP, int* csplit, int* maxNode, int* minbucket, int* minsplit, int* nIter, int* nTrees, int* pMutateMajor, int* pMutateMinor,int *pCrossover, int *pSplit, int *pPrune, int* method, double* alpha, int* seed){
	container = new Container(nInst, nVar, varType, ndata, weights, prediction, splitV, splitP, csplit, maxNode, minbucket, minsplit, nIter,nTrees, pMutateMajor, pMutateMinor, pCrossover, pSplit, pPrune, method, alpha, seed);
}//tree

void freememory(void){
    delete container;
    container = NULL;
}// cleanup
}//extern C 

static void chkIntFn(void *dummy) {
     R_CheckUserInterrupt();
}

bool Container::checkInterrupt(void){
  return (R_ToplevelExec(chkIntFn, NULL) == FALSE);
}

Container::Container(int* nInstances, int* nVariables, int *varType, double* ndata, int* weights, int* prediction, int *splitV, double *splitP, int* csplit, int* maxNode, int *minbucket,int* minsplit, int* nIter, int* nTrees, int* pMutateMajor, int* pMutateMinor, int* pCrossover, int* pSplit, int* pPrune, int* method, double* alpha, int* seed ){
    // constructor
    if(*seed < 0)
	    *seed = (unsigned)time(0);
    srand(*seed);
    this->maxNode = *maxNode;
    this->minsplit = *minsplit;
    this->nTrees = *nTrees;
    this->nInstances = *nInstances;
    this->nIterations = *nIter;
    this->nVariables = *nVariables;
    this->minbucket = *minbucket;
    this->minsplit = *minsplit;
    this->variables = new variable*[this->nVariables];
    this->data = new double*[this->nInstances];
    this->weights = new int[this->nInstances];
    this->performanceHistory = new double[50];
    this->probMutateMajor = *pMutateMajor;
    this->probMutateMinor = *pMutateMinor + this->probMutateMajor;
    this->probSplit = *pSplit + this->probMutateMinor;
    this->probPrune = *pPrune + this->probSplit;
    this->probCrossover = *pCrossover + this->probPrune;
    this->elitismRange = max((int)ceil((double)(this->nTrees/20.0)),2);
    this->nTrees +=  this->elitismRange;
    this->method = *method;
    this->elitismList = new int[this->elitismRange];

    for(int i = 0; i < this->elitismRange; i++)
        this->elitismList[i] = 999999;

    // transform data in vector format into a n x p matrix
    this->sumWeights = 0;
    for (int i = 0; i < this->nInstances; i++){
        this->data[i] = new double[this->nVariables];
        this->weights[i] = weights[i];
        sumWeights += this->weights[i];
    }
    for(int v = 0; v < this->nVariables; v++){
        for(int i = 0; i < this->nInstances; i++){
            data[i][v] = ndata[v*this->nInstances+i];
        }
    }
    // calculate variance of the dependend variable for regression trees
    if(this->method == 6){
        double mean = 0;
        double squaredSum = 0;

        for(int i = 0; i < this->nInstances; i++){
                 mean += data[i][this->nVariables-1] * this->weights[i];
                 squaredSum += data[i][this->nVariables-1]*data[i][this->nVariables-1]*this->weights[i];
        }
        mean /= ((double) sumWeights);

        this->populationMSE = ((double) sumWeights) * (1.0 / ((double) sumWeights) * squaredSum - (mean * mean));
    }

    for(int i = 0; i < 50; i++){
        this->performanceHistory[i] = 999999999;
    }
    this->initVariables(varType);

    this->maxCat = 1;
    for(int i = 0; i < (this->nVariables-1); i++)
        if( varType[i] < -this->maxCat )
            this->maxCat = -varType[i];

    this->alpha = (*alpha);
    this->trees = new Tree*[this->nTrees];

    for(int i = 0; i < this->nTrees; i++){
        this->trees[i] = new Tree(&this->nInstances, &this->nVariables, this->data, this->weights, &this->maxCat, this->variables, &this->maxNode, &this->minbucket, &this->minsplit);
    }

    bool allTreesInitialized = TRUE;
    for(int i = 0; i < this->nTrees; i++){
        if(this->trees[i]->splitV[0] < 0)
        	allTreesInitialized = FALSE;
        else
       		this->evaluateTree(i, true, 0);
    }

    if(allTreesInitialized == TRUE){
        // start evolving the initial solution    
    	bool succ =  this->evolution();

    	// write the information of the best tree into the variables passed from R
         if(succ == true){
   	 	if(this->elitismList[0] < this->nTrees){
              		*nIter = this->nIterations;
       	      		for(int i = 0; i < *this->trees[this->elitismList[0]]->maxNode; i++){
		              if(this->trees[this->elitismList[0]]->splitV[i] >= 0 ){
            			   splitV[i] = this->trees[this->elitismList[0]]->splitV[i]+1;
			           splitP[i] = this->trees[this->elitismList[0]]->splitP[i];

                	    	   for(int k = 0; k < this->maxCat; k++){
               			 	if( variables[*this->trees[this->elitismList[0]]->nodes[i]->splitV]->isCat == true
	           			 && k < variables[*this->trees[this->elitismList[0]]->nodes[i]->splitV]->nCats ){
        	                	 	csplit[i* this->maxCat + k] = this->trees[this->elitismList[0]]->csplit[k][i];
                	   		}else if(this->maxCat > 1){
                        			csplit[i* this->maxCat + k] = 2;
                         		}
                    	    	   }
             		      }else{
		                      splitV[i] = -999999;
        	   		      splitP[i] = -999999;

                		       // if there is only one independend variabe, and the variable is numeric, csplit
		            	       // is not a vector. in this cas csplit is not used.
			      	       if(this->maxCat > 1){
		               	       	      for(int k = 0; k < this->maxCat; k++){
                			             csplit[i * this->maxCat + k] = 2;
		                	     }
                	      		}
              		      }
           		}
           	 	for(int i = 0; i < this->nInstances; i++){
                       		prediction[i] = this->trees[this->elitismList[0]]->classification[i];
                 	}
       	    } // if elitism
       } // if succ
   }// if allTreeInitialized
} // end Container


bool Container::evolution(){
    // evolves the initial solution
    double evalValue; // stores the return value from variation operators
    int randomNumber = 0; // random number calculated for each tree in each iteration. used for operator selection
    bool elitismFlag = false;
    for(int i = 0; i < this->nIterations; i++){
        // checks for user interupts via control-c
        
      // your code somewhere ...
	if (checkInterrupt()) { 
	// user interrupted ... 
		return false;
	}
        
        
        R_CheckUserInterrupt();
        for(int j = 0; j < nTrees; j++){
            // check if tree j is in the elitism list
            // if it is; with a small probability a copy of tree j is placed in the population
            if(i > 10 && this->elitismList[this->elitismRange-1] < this->nTrees){
                elitismFlag = false;
                for(int f = 0; f < this->elitismRange &&  elitismFlag == false; f++){
                    if( this->elitismList[f] == j){
                        elitismFlag = true;
                    }
                }

                if (elitismFlag == true){
                    int m = this->getGenitor();
                    int rn = rand()%1000;
                    if( rn < 20 && this->elitismList[0] < this->nTrees ){
                        if(this->trees[m]->performance/this->trees[ this->elitismList[0] ]->performance < 1.03 ){
                             m = this->getRandomTree(false);
                        }
                        this->overwriteTree(j, m);
                    }
                }
           }
           // call of variation operators
           if(elitismFlag == false){
               randomNumber= (rand()%100);
              if(randomNumber < this->probMutateMajor){
                     evalValue = this->initMutateNode(j, false);
              }else if(randomNumber < this->probMutateMinor){
                     evalValue = this->initMutateNode(j, true); 
              }else if(randomNumber < this->probSplit){
                     evalValue = this->splitNode(j);
              }else if(randomNumber < this->probPrune){
                    evalValue = this->pruneNode(j);
              }else{
                    evalValue = this->crossover(j);            
              }
               // if j belongs in the elitism list an extra copy is made replacing a random tree
               if( i > 10){
                     if(this->updatePerformanceList(j) == true){
                         this->overwriteTree(j, this->getRandomTree(false));
                     }
               }else if(i > 3){
                   this->updatePerformanceList(j);
               }
           }// end if(elitismFlag == false)
      }// end for(int j=0; j<nTrees; j++)

      if(i > 7 && i%10 == 0 && this->elitismList[this->elitismRange-1] < this->nTrees){
          int pos = (i/10)%50;
          this->performanceHistory[pos] = 0;
          for(int nt = 0; nt < this->elitismRange; nt++){
               this->performanceHistory[pos] += this->trees[this->elitismList[nt]]->performance;
          }
          if(i > 1000){ // check if the population has converged
              int nextpos = 0;
              if(pos != 49)
                  nextpos = pos+1;
              if(this->method == 6){
                  if( (this->performanceHistory[pos] - this->performanceHistory[nextpos])  >= 0){  
                      this->nIterations = i;
                  }
              }else{
                  if(this->performanceHistory[pos] / this->performanceHistory[nextpos] >= 0.9995 )
                     this->nIterations = i;
              }
          }
      }
   }
    // pruning of all trees 
    for(int i = 0; i < this->nTrees; i++)
        this->pruneAllNodes(i);
    return true;
} // end evolution


double Container::pruneNode(int treeNumber){
        // variation operator prune
        // selects a random node and prunes it
    if(this->trees[treeNumber]->nNodes > 2){
        double oldPerformance = this->trees[treeNumber]->performance;
	int nodeNumber = 0;
	bool flag = false;
	for(int i = 0; i < 10 && flag == false; i++){
            nodeNumber = randomSplitNode(treeNumber);
            if(this->trees[treeNumber]->splitV[nodeNumber*2+1] < 0 &&
                    this->trees[treeNumber]->splitV[nodeNumber*2+2] < 0)
                flag = true;
	}
        int parent= (int) floor((nodeNumber-1)/2) ;
        if(flag == false || parent < 0)
            return -1;

	int oldSplitV = this->trees[treeNumber]->splitV[nodeNumber];
	double oldSplitP = this->trees[treeNumber]->splitP[nodeNumber];
	this->trees[treeNumber]->splitV[nodeNumber] = -999999;
	this->trees[treeNumber]->splitP[nodeNumber] = -999999;
	if(nodeNumber % 2 == 0 ) {
            this->trees[treeNumber]->nodes[parent]->rightChild = NULL;
	}else{
            this->trees[treeNumber]->nodes[parent]->leftChild = NULL;
	}
	this->trees[treeNumber]->nNodes--;

        if( this->evaluateTree(treeNumber, false, parent) == false){
                // cout << "warning: invalid tree is replaced by a random tree (1)" << endl;
                this->overwriteTree(treeNumber);
            return -5;
        }

        int accept = this->evaluateNewSolution(treeNumber, &oldPerformance);

        if(accept >= 0){// node is pruned
            delete this->trees[treeNumber]->nodes[nodeNumber];
            this->trees[treeNumber]->nodes[nodeNumber] = NULL;
            if( this->evaluateTree(treeNumber, false, parent) == false){
                // cout << "warning: invalid tree is replaced by a random tree (2)" << endl;
                this->overwriteTree(treeNumber);
                return -5;
            }
            return accept;
        }else if(accept == -1){ // reverse prune due to performance reasons
            this->trees[treeNumber]->nNodes++;
            if(nodeNumber%2 == 0 ) {// add Child back to parent
                    this->trees[treeNumber]->nodes[(int) floor((nodeNumber-1) / 2)]->rightChild = this->trees[treeNumber]->nodes[nodeNumber];
            }else{
                    this->trees[treeNumber]->nodes[(int) floor((nodeNumber-1) / 2)]->leftChild = this->trees[treeNumber]->nodes[nodeNumber];
            }
            this->trees[treeNumber]->splitV[nodeNumber] = oldSplitV;
            this->trees[treeNumber]->splitP[nodeNumber] = oldSplitP;
            if( this->evaluateTree(treeNumber, false, parent) == false){
                // cout << "warning: invalid tree is replaced by a random tree (3)" << endl;
                this->overwriteTree(treeNumber);
                return -5;
            }
            return -1;
	}else{
            // cout << "warning: invalid tree is replaced by a random tree (4) " << endl;
            this->overwriteTree(treeNumber);
            return -2;
        }

    }else return -4;
}// end pruneNode


int Container::pruneAllNodes(int treeNumber){ 
    // sequentelly tries to prune every node in the tree
    double oldPerformance;
    bool flag = false;
    if( this->elitismList[0] == treeNumber )
        return 0;
    oldPerformance = this->trees[treeNumber]->performance;
        if(this->trees[treeNumber]->nNodes > 2){
            for(int nodeNumber = 1; nodeNumber < this->maxNode; nodeNumber++){
                int parent = (int) floor((nodeNumber-1)/2);
                if( this->trees[treeNumber]->splitV[nodeNumber] >= 0 &&
                    this->trees[treeNumber]->splitV[nodeNumber*2+1] < 0 &&
                    this->trees[treeNumber]->splitV[nodeNumber*2+2] < 0 && parent >= 0){
                    for(int f = 0 ; f < this->elitismRange; f++){
                         if( this->elitismList[f] == treeNumber ){
                             int target = this->getGenitor();
                             if(target == treeNumber)
                                 return -1;
                             this->overwriteTree(treeNumber,target);
                             this->pruneAllNodes(target);
                             return 0;
                         }
                    }
                    int oldSplitV = this->trees[treeNumber]->splitV[nodeNumber];
                    double oldSplitP = this->trees[treeNumber]->splitP[nodeNumber];
                    this->trees[treeNumber]->splitV[nodeNumber] = -999999;
                    this->trees[treeNumber]->splitP[nodeNumber] = -999999;
                    if(nodeNumber%2 == 0) {
                        this->trees[treeNumber]->nodes[parent]->rightChild = NULL;
                    }else{
                        this->trees[treeNumber]->nodes[parent]->leftChild = NULL;
                    }
                    this->trees[treeNumber]->nNodes--;
                    if( this->evaluateTree(treeNumber, false, parent) == false){
                        // cout << "warning: invalid tree is replaced by a random tree (5)" << endl;
                        this->overwriteTree(treeNumber);
                        return -5;
                    }else{
                        // pruning was successful
                        if( this->trees[treeNumber]->performance < oldPerformance){
                            delete this->trees[treeNumber]->nodes[nodeNumber];
                            this->trees[treeNumber]->nodes[nodeNumber] = NULL;
                            if( this->evaluateTree(treeNumber, false, parent) == false){
                                // cout << "warning: invalid tree is replaced by a random tree (7)" << endl;
                                this->overwriteTree(treeNumber);
                                return -5;
                            }
                            flag = true;
                            this->updatePerformanceList(treeNumber);
                        }else{ // reverse prune due to performance reasons
                            this->trees[treeNumber]->nNodes++;
                            if(nodeNumber%2 == 0 ) {// add Child to parent
                                 this->trees[treeNumber]->nodes[parent]->rightChild = this->trees[treeNumber]->nodes[nodeNumber];
                            }else{
                                 this->trees[treeNumber]->nodes[parent]->leftChild = this->trees[treeNumber]->nodes[nodeNumber];
                            }
                            this->trees[treeNumber]->splitV[nodeNumber] = oldSplitV;
                            this->trees[treeNumber]->splitP[nodeNumber] = oldSplitP;
                            if( this->evaluateTree(treeNumber, false, parent) == false){
                                // cout << "warning: invalid tree is replaced by a random tree (8)" << endl;
                                this->overwriteTree(treeNumber);
                                return -5;
                            }
                        }
                    }
                }
            }
        }
    if(flag == true)
        this->pruneAllNodes(treeNumber);
    return 1;
} // end pruneAllNodes


double Container::splitNode(int treeNumber){
        // assigns a new random split to a random node       
        bool flagValid = false;
	int terminalNode = -1;
        double oldPerformance = this->trees[treeNumber]->performance;
        int s_const = this->nVariables;
        int t_const = 10;
   	for(int t = 0; t < t_const && flagValid == false; t++){ // a maximum of t_const terminal nodes are tryed to split
            terminalNode = this->randomTerminalNode(treeNumber);
            if(terminalNode <= 0)
                 return -2;
            if(t == 0)
              this->trees[treeNumber]->nNodes++;
            if(terminalNode < this->maxNode ){ // check if the added split exceets the maximum tree depth
                  for(int s = 0; s < s_const && flagValid == false; s++){// a maximum of s_const split variables are tried
                    this->randomSplitVariable(treeNumber, terminalNode);
                    if(variables[this->trees[treeNumber]->splitV[terminalNode]]->isCat == false  // to few observations to split at this node
                          &&  this->randomSplitPoint(treeNumber, terminalNode) == false){
                        s = s_const;             
                    }else{
                        if(s == 0)
                             this->trees[treeNumber]->initNode(terminalNode);
                             
                        if(variables[this->trees[treeNumber]->splitV[terminalNode]]->isCat == true){
                            this->trees[treeNumber]->randomizeCategories(terminalNode);
                        }

                        if(terminalNode % 2 == 0 ){ // add child to parent
                                this->trees[treeNumber]->nodes[(int) floor((terminalNode-1)/2) ]->rightChild = this->trees[treeNumber]->nodes[terminalNode];
                        }else{
                                this->trees[treeNumber]->nodes[(int) floor((terminalNode-1)/2) ]->leftChild = this->trees[treeNumber]->nodes[terminalNode];
                        }
                        flagValid = this->evaluateTree(treeNumber, false, (int) floor((terminalNode-1)/2) ) ;
                    }
                }
            }
            
            if(flagValid == false){
               if(this->trees[treeNumber]->nodes[terminalNode] != NULL){
                  if(this->trees[treeNumber]->deleteChildNodes(terminalNode) == false ){
                      // cout << "warning: invalid tree is replaced by a random tree (9)" << endl;
                      this->overwriteTree(treeNumber);
                      return -10;
                  }
                  this->trees[treeNumber]->nNodes++; 
               }else{
                      this->trees[treeNumber]->splitV[terminalNode] = -999999;
               }
               if(this->evaluateTree(treeNumber, false , (int) floor((terminalNode - 1) / 2) ) == false) {
                      // cout << "warning: invalid tree is replaced by a random tree (10) " << endl;
                      this->overwriteTree(treeNumber);
                      return -10;
               }
            }
	}
       	if(flagValid == false){
            this->trees[treeNumber]->nNodes--;
            return -4;
        }
        int accept = this->evaluateNewSolution(treeNumber, &oldPerformance);
        if(accept >= 0){
            return accept;
        }else if(accept == -1){
            if(this->trees[treeNumber]->deleteChildNodes(terminalNode) == false){
                // cout << "warning: invalid tree is replaced by a random tree (12)" << endl;
                this->overwriteTree(treeNumber);
                return -5;
            }
            if( this->evaluateTree(treeNumber, false, (int)((terminalNode-1)/2)) == false){
                // cout << "warning: invalid tree is replaced by a random tree (13)" << endl;
                this->overwriteTree(treeNumber);
                return -5;
            }
            return -1;
        }else{
            this->overwriteTree(treeNumber);
            return -2;
        }
        return -5;
} // splitNode


int Container::randomTerminalNode(int treeNumber){
    // selects and returns a random terminal node
    int* nodes = new int[ trees[treeNumber]->nNodes + 1 ];
    int j = 0;
    for(int i = 0; i < this->maxNode && j < trees[treeNumber]->nNodes; i++){
        if(i*2+1 < this->maxNode){
            if(trees[treeNumber]->splitV[i] >= 0 &&  trees[treeNumber]->splitV[i*2+1] < 0){
                nodes[j] = i*2+1;
                j++ ;
            }
        }if(i*2+2 < this->maxNode){
            if(trees[treeNumber]->splitV[i] >= 0 &&  trees[treeNumber]->splitV[i*2+2] < 0){
                nodes[j] = i*2+2;
                j++ ;
            }
        }
    }
    bool flag = false;
    int randomNode = -1;
    if(j == 0){
        delete [] nodes;
        nodes = NULL;
        return -1;
    }
    for( int i = 0; flag == false && i < 101; i++){
        randomNode= rand()%j;
        if( trees[treeNumber]->nodes[(int) ((nodes[randomNode]-1)/2)]->sumLocalWeights > this->minsplit){
            flag = true;
        }
    }
    if(flag == false){
        delete [] nodes;
        nodes = NULL;
        return -1;
    }else{
        int returnvalue = nodes[randomNode];
        delete [] nodes;
        nodes = NULL;
        return returnvalue;
    }
} // randomTerminalNode


double Container::initMutateNode(int treeNumber, bool isMinorChange){
        // select a random node and initializes the mutationNode()
        int changedNode = this->randomSplitNode(treeNumber);
        double accept = this->mutateNode(treeNumber, changedNode, isMinorChange);
        for(int i = 0; i < 3 && accept == -5; i++){
            changedNode = this->randomSplitNode(treeNumber);
            accept = this->mutateNode(treeNumber, changedNode, isMinorChange);
        }
        if( this->evaluateTree(treeNumber, false, 0) == false){
            // cout << "warning: invalid tree is replaced by a random tree (14)" << endl;
            this->overwriteTree(treeNumber);
            return -5;
        }
        return accept;
} // initMutateNode


double Container::mutateNode(int treeNumber, int nodeNumber, bool isMinorChange){
        // variation operator mutateNode
        double oldPerformance = this->trees[treeNumber]->performance;
        int oldSplitV = this->trees[treeNumber]->splitV[nodeNumber];  // for reversal in case no split can be found for this node
        int oldSplitV2 = -999999;
	double oldSplitP = this->trees[treeNumber]->splitP[nodeNumber];
        int *oldCsplit = new int[this->maxCat];
        if(variables[this->trees[treeNumber]->splitV[nodeNumber] ]->isCat == true){
            for(int i = 0; i < variables[this->trees[treeNumber]->splitV[nodeNumber]]->nCats; i++){
                oldCsplit[i] = this->trees[treeNumber]->csplit[i][nodeNumber];
            }
            for(int i = variables[ this->trees[treeNumber]->splitV[nodeNumber]]->nCats; i < this->maxCat; i++){
                oldCsplit[i] = 2;
            }
        }
        Node* tempNode = NULL;
        int returnValue = 0;
        int parent = 0;
        int noOfPrunedNodes = 0;
        if(isMinorChange == false){// change split-variable and split-point; non valid nodes are pruned later
            if( (rand()%2) == 1){     // Change split Variable?
                 this->randomSplitVariable(treeNumber, nodeNumber);
            }
            if(variables[this->trees[treeNumber]->splitV[nodeNumber]]->isCat == false){
                this->randomSplitPoint(treeNumber, nodeNumber);
            }else{
                 this->trees[treeNumber]->randomizeCategories(nodeNumber);
            }
        }else{ // only change split-point by a minor degree. several attempts to find a valid tree
            if(variables[this->trees[treeNumber]->splitV[nodeNumber]]->isCat == false){
                this->changeSplitPoint(treeNumber, nodeNumber);
                for(int i = 0; i < 5 &&  this->evaluateTree(treeNumber, false, nodeNumber) == false; i++ ){
                     this->trees[treeNumber]->splitP[nodeNumber] = oldSplitP;
                     this->changeSplitPoint(treeNumber, nodeNumber);
                }
            }else{
                this->changeRandomCategories(treeNumber, nodeNumber);
                for(int i = 0; i < 5 &&  this->evaluateTree(treeNumber, false, nodeNumber) == false; i++){
                    this->changeRandomCategories(treeNumber, nodeNumber);
                }
            }
        }
        returnValue = this->trees[treeNumber]->predictClass(this->minbucket, this->minsplit, false,  nodeNumber) ;
        if(returnValue == 0 || returnValue == nodeNumber){  // reverse mutation
            this->trees[treeNumber]->splitV[nodeNumber] = oldSplitV;
            this->trees[treeNumber]->splitP[nodeNumber] = oldSplitP;
            if(variables[ oldSplitV ]->isCat == true){
                for(int i = 0; i < variables[oldSplitV ]->nCats; i++){
                    this->trees[treeNumber]->csplit[i][nodeNumber] = oldCsplit[i];
                }
            }
            delete [] oldCsplit;
            oldCsplit = NULL;
            if(this->evaluateTree(treeNumber, false, nodeNumber ) == false){
                // cout << "warning: invalid tree is replaced by a random tree (15)" << endl;
                this->overwriteTree(treeNumber);
            }
            return -5;
        }

        if(returnValue != -1){ // some pruning is necessary (not enough observations in the terminal nodes)
            noOfPrunedNodes = calculateNoOfNodesInSubtree(treeNumber, returnValue);
            oldSplitV2 =  this->trees[treeNumber]->splitV[returnValue];
            this->trees[treeNumber]->splitV[returnValue] = -999999;
            this->trees[treeNumber]->nNodes -= noOfPrunedNodes;
            parent = (int) ((returnValue-1)/2);
            tempNode = this->trees[treeNumber]->nodes[returnValue];
            if(returnValue % 2 == 0){
               this->trees[treeNumber]->nodes[parent]->rightChild = NULL;
            }else{
               this->trees[treeNumber]->nodes[parent]->leftChild = NULL;
            }
        }
        // case that both sides of the mutated node have to be pruned
        // mutation is reversed in any case
        if(this->evaluateTree(treeNumber, false, nodeNumber ) == false){
           this->trees[treeNumber]->splitV[nodeNumber] = oldSplitV;
           this->trees[treeNumber]->splitP[nodeNumber] = oldSplitP;
           if(variables[ oldSplitV ]->isCat == true){
              for(int i = 0; i < variables[oldSplitV ]->nCats; i++){
                  this->trees[treeNumber]->csplit[i][nodeNumber] = oldCsplit[i];
              }
           }
           if(returnValue%2 == 0)
              this->trees[treeNumber]->nodes[parent]->rightChild = tempNode;
           else
               this->trees[treeNumber]->nodes[parent]->leftChild = tempNode;
           tempNode = NULL;
           this->trees[treeNumber]->splitV[returnValue] = oldSplitV2;
           this->trees[treeNumber]->nNodes += noOfPrunedNodes;
           delete [] oldCsplit;
           oldCsplit = NULL;

           if(this->evaluateTree(treeNumber, false, nodeNumber ) == false){
                // cout << "warning: invalid tree is replaced by a random tree (16)" << endl;
                this->overwriteTree(treeNumber);
           }
           return -5;
        }

        int evaluation= this->evaluateNewSolution(treeNumber, &oldPerformance);

        if(evaluation < 0){ // reverse due to performance reasons
            this->trees[treeNumber]->splitV[nodeNumber] = oldSplitV;
            this->trees[treeNumber]->splitP[nodeNumber] = oldSplitP;
            if(variables[ oldSplitV ]->isCat == true){
                for(int i = 0; i < variables[oldSplitV ]->nCats; i++){
                    this->trees[treeNumber]->csplit[i][nodeNumber] = oldCsplit[i];
                }
            }
        }

        delete [] oldCsplit;
        oldCsplit = NULL;

        if(returnValue == -1){ // all nodes of the new tree are valid
            this->evaluateTree(treeNumber, false, nodeNumber );
            return evaluation;
        }else{ // case that there are invalid nodes in the new tree that have to be removed
            // first put back the removed subtree that was invalid
            if(returnValue % 2 == 0)
               this->trees[treeNumber]->nodes[parent]->rightChild = tempNode;
            else
               this->trees[treeNumber]->nodes[parent]->leftChild = tempNode;
            tempNode = NULL;
            this->trees[treeNumber]->splitV[returnValue] = oldSplitV2;
            this->trees[treeNumber]->nNodes += noOfPrunedNodes;

            // the subtree is pruned
            if(evaluation >= 0 ){
                this->trees[treeNumber]->deleteChildNodes(returnValue);
                if(this->evaluateTree(treeNumber, false, nodeNumber) == false){
                // cout << "warning: invalid tree is replaced by a random tree (17)" << endl;
                    this->overwriteTree(treeNumber);
                    return -5;
                }
                return evaluation;
            }else{ // mutation was not accepted; tree stays the same; in this case splitV and splitP alread have been reversed earlier
                if(this->evaluateTree(treeNumber, false, nodeNumber) == false){
                // cout << "warning: invalid tree is replaced by a random tree (18)" << endl;
                    this->overwriteTree(treeNumber);
                    return -5;
                }
                return evaluation;
            }
        }
        return -5
;}//end mutateNode


int Container::calculateNoOfNodesInSubtree(int treeNumber, int nodeNumber){
    // used by mutateNode to calculate the number of nodes in a subtree below nodeNumber
    if(this->trees[treeNumber]->splitV[nodeNumber*2+1] >= 0  && this->trees[treeNumber]->nodes[nodeNumber]->leftChild != NULL)
         return calculateNoOfNodesInSubtree(treeNumber, nodeNumber*2+1)+1;
    else return 1;
    if(this->trees[treeNumber]->splitV[nodeNumber*2+2] >= 0 && this->trees[treeNumber]->nodes[nodeNumber]->rightChild != NULL)
         return calculateNoOfNodesInSubtree(treeNumber, nodeNumber*2+2)+1;
    else return 1;
} // calculateNoOfNodesInSubtree


bool Container::changeRandomCategories(int treeNumber, int nodeNumber){
    // used by mutate for minor changes of categories
    int left = 0;
    int right = 0;
    if(variables[ this->trees[treeNumber]->splitV[nodeNumber] ]->nCats < 3)
        return false;

    for(int i = 0; i < this->variables[ *this->trees[treeNumber]->nodes[nodeNumber]->splitV ]->nCats; i++){
        if(this->trees[treeNumber]->csplit[i][nodeNumber] == 1)
            left++;
        else if(this->trees[treeNumber]->csplit[i][nodeNumber] == 3)
            right++;
        else{
            if(rand()%2 == 1){
               this->trees[treeNumber]->csplit[i][nodeNumber] = 1;
               left++;
            }else{
               this->trees[treeNumber]->csplit[i][nodeNumber] = 3;
               right++;
            }
        }
    }
    int changes = max(1, rand()%( (int) variables[ this->trees[treeNumber]->splitV[nodeNumber] ]->nCats/10+1) );
    int i = 0;
    int changedCat;
     while(changes > 0 && i < changes*3){
        changedCat = rand()%this->variables[ *this->trees[treeNumber]->nodes[nodeNumber]->splitV]->nCats;
        if(this->trees[treeNumber]->csplit[changedCat][nodeNumber] == 1 && left > 1){
            this->trees[treeNumber]->csplit[changedCat][nodeNumber] = 3;
            left--;
            right++;
            changes--;
        }else if(this->trees[treeNumber]->csplit[changedCat][nodeNumber] == 3 && right > 1){
            this->trees[treeNumber]->csplit[changedCat][nodeNumber] = 1;
            left++;
            right--;
            changes--;
        }
        i++;
    }
    return true;
} // end changeRandomCategories


int Container::getRandomTree(bool elitism){
   int temp = this->elitismList[rand()%this->elitismRange];
    if(elitism == true && temp >= 0 && temp < this->nTrees){
        return temp;
    }else{
        bool elitismFlag = false;
        int randTree = rand()%this->nTrees;
        while(elitismFlag == false && this->elitismList[this->elitismRange-1] < this->nTrees){
            elitismFlag = true;
            for(int f=0 ;f < this->elitismRange && elitismFlag == true; f++){
                if( this->elitismList[f] == randTree ){
                    elitismFlag = false;
                }
            }
            if(randTree == this->elitismList[0])
                    elitismFlag = false;
            if (elitismFlag == false)
               randTree = rand()%this->nTrees;
        }
	return randTree;
   }
} // end get Trandom Tree


int Container::getGenitor(){
    int genitor = 0;
    if (this->elitismList[0] == 0)
        genitor = 1;
    for(int i = genitor+1; i < this->nTrees; i++){
        if(this->trees[i]->performance > this->trees[genitor]->performance){
            if(i != this->elitismList[0])
                 genitor = i;
        }
    }
    return genitor;
}


void Container::randomSplitVariable(int treeNumber, int nodeNumber){
    // used by splitNode() and mutateNode()
        this->trees[treeNumber]->splitV[nodeNumber] = (rand()%(this->nVariables-1));
}


bool Container::randomSplitPoint(int treeNumber, int nodeNumber){
     // assigns a random split points
     // used by splitNode() and mutateNode() if the isMinor is set to false
    if(this->variables[abs(this->trees[treeNumber]->splitV[nodeNumber])]->isCat == false){

        int localInstances = 0;
        if(nodeNumber%2 == 0 ){ // add Child to parent and remove number of terminal instances
             localInstances = this->trees[treeNumber]->nodes[(int) floor((nodeNumber-1)/2) ]->sumRightLocalWeights;
        }else{
             localInstances = this->trees[treeNumber]->nodes[(int) floor((nodeNumber-1)/2) ]->sumLeftLocalWeights;
        }
        if(localInstances < this->minsplit)
            return false;

        double min = 1;
        double max = (this->variables[abs(this->trees[treeNumber]->splitV[nodeNumber]) ]->nCats)-1;
        int randomSplitPoint = 0;
        double randomNumber = 0.0;

        for(int k = 0; k < 10 && (randomSplitPoint < min || randomSplitPoint > max); k++ ){
            randomNumber = 0.0;
            for(int i = 0; i < 12; i++){
         	   randomNumber += ((rand()%1000)+1) / 1000.0;
            }
            randomSplitPoint = (int)round(((randomNumber-6)*(max-min)/2.0)+( (max+min)/2.0 ));
        }      
       
        if(randomSplitPoint < min || randomSplitPoint > max){
              randomSplitPoint = (int)round((min+max)/2.0);
        }

        this->trees[treeNumber]->splitP[nodeNumber] = this->variables[ abs(this->trees[treeNumber]->splitV[nodeNumber]) ]->sortedValues[randomSplitPoint] ;
        return true;
    }
    this->trees[treeNumber]->splitP[nodeNumber] = -999999;
    return true;
} // end getSplitPoint


bool Container::changeSplitPoint(int treeNumber, int nodeNumber){
        // used by mutate() to change split point by a minor degree (<= 10% of the range of values)
        double mini = 1;
        double maxi = (this->variables[ this->trees[treeNumber]->splitV[nodeNumber] ]->nCats)-1;
        if((maxi - mini) < 2)
            return false;

        int oldPos = 0;
        bool flag = false;
        for(int i = 0; i < (this->variables[ this->trees[treeNumber]->splitV[nodeNumber] ]->nCats) && flag == false; i++){
             if(this->trees[treeNumber]->splitP[nodeNumber] == this->variables[ abs(this->trees[treeNumber]->splitV[nodeNumber]) ]->sortedValues[i]){
                oldPos = i;
                flag = true;
             }
        }

        int randomNumber=  max(1, (rand()%( (int)variables[ this->trees[treeNumber]->splitV[nodeNumber] ]->nCats/10+1))  );
        if(rand()%2 == 0)
               ;
        else
              randomNumber = -randomNumber;

        if(randomNumber > 0 && oldPos+randomNumber > maxi)
            randomNumber = -randomNumber;
        else if(randomNumber < 0 && oldPos+randomNumber < mini)
            randomNumber = -randomNumber;

        
        int randomSplitPoint= oldPos+randomNumber;

        if(randomSplitPoint < mini || randomSplitPoint > maxi){
            randomSplitPoint = (int)round((mini+maxi)/2.0);
        }
        this->trees[treeNumber]->splitP[nodeNumber] = this->variables[ abs(this->trees[treeNumber]->splitV[nodeNumber]) ]->sortedValues[randomSplitPoint];
    return true;
} // end changeSplitPoint


int Container::randomSplitNode(int treeNumber){
	// returns an random internal node for splitting
	int* nodes= new int[ trees[treeNumber]->nNodes ];
	int j = 0;
	for(int i = 0; i < this->maxNode && j < this->trees[treeNumber]->nNodes; i++){
            if(trees[treeNumber]->splitV[i] >= 0){
                nodes[j] = i;
                j++ ;
            } 
	}
        int rvalue = nodes[rand()%(j)];
        delete [] nodes;
        nodes = NULL;
        return rvalue;
} // end randomSplitNode


bool Container::evaluateTree(int treeNumber, bool pruneIfInvalid, int nodeNumber){
        // calculate predictions and check tree validity
	if( this->trees[treeNumber]->predictClass( this->minbucket, this->minsplit, pruneIfInvalid, nodeNumber ) != -1){
            return false;
	}else{
            int found = 0;
            for(int i = nodeNumber; i < this->maxNode && found < this->trees[treeNumber]->nNodes; i++){
                if(this->trees[treeNumber]->splitV[i] >= 0){
                    found++;                                  
                    if(this->trees[treeNumber]->nodes[i]->sumLeftLocalWeights == 0 &&  this->trees[treeNumber]->nodes[i]->sumRightLocalWeights == 0){
                   ;
                    }else if(this->trees[treeNumber]->nodes[i]->sumLeftLocalWeights >= this->minbucket &&  this->trees[treeNumber]->splitV[i*2+2] >= 0){
                    ;
                    }else if(this->trees[treeNumber]->nodes[i]->sumRightLocalWeights >= this->minbucket  &&  this->trees[treeNumber]->splitV[i*2+1] >= 0){
                    ;
                    }else if(this->trees[treeNumber]->nodes[i]->sumLeftLocalWeights <  this->minbucket
                            ||  this->trees[treeNumber]->nodes[i]->sumRightLocalWeights < this->minbucket
                            ||  this->trees[treeNumber]->nodes[i]->sumLeftLocalWeights + this->trees[treeNumber]->nodes[i]->sumRightLocalWeights < this->minsplit ){
                        return false;
                    }
                }
            }
	}
	this->trees[treeNumber]->calculateTotalCosts(this->method, this->alpha, this->sumWeights, this->populationMSE);
	return true;
} // end evaluateTree


Container::~Container(){
	for (int i = 0; i < this->nTrees; i++) 
            delete trees[i];
	delete [] trees;
        trees = NULL;
	for (int i = 0; i < this->nInstances; i++) {
            delete [] data[i];
        }
        delete [] data;
        data = NULL;

	for (int i = 0; i < this->nVariables; i++) {
            if(variables[i] != NULL)
                    delete variables[i];
	}
	delete [] variables;
        variables = NULL;
        delete [] performanceHistory;
        performanceHistory = NULL;
        delete [] elitismList;
        elitismList = NULL;
        delete [] weights;
        weights = NULL;
} // end ~Container


void Container::initVariables(int* varType){
    for (int i = 0; i < this->nVariables ; i++){
        this->variables[i] = new variable( i, this->nVariables-1, this->nInstances, this->data, varType[i]);
    }
} // end initVariables


void Container::overwriteTree(int targetPos){
    // replaces the tree with treenumber targetPos by a random tree
    delete this->trees[targetPos];
    this->trees[targetPos] = NULL;
    int sourcePos;
    sourcePos = this->getRandomTree(true);
    while(sourcePos == targetPos)
        sourcePos= this->getRandomTree(true);
    this->trees[targetPos] = new Tree(&this->nInstances, &this->nVariables, this->data, this->weights, this->trees[sourcePos]->splitV, this->trees[sourcePos]->splitP, this->trees[sourcePos]->csplit, &this->maxCat, &this->trees[sourcePos]->nNodes, this->variables, &this->maxNode);
    while( this->evaluateTree(targetPos, false, 0) == false){
        delete this->trees[targetPos];
        this->trees[targetPos] = NULL;
        while(sourcePos == targetPos)
              sourcePos = this->getRandomTree(true);
        this->trees[targetPos] = new Tree(&this->nInstances, &this->nVariables, this->data,  this->weights, this->trees[sourcePos]->splitV, this->trees[sourcePos]->splitP, this->trees[sourcePos]->csplit, &this->maxCat, &this->trees[sourcePos]->nNodes, this->variables, &this->maxNode);
    }
} // end overwriteTree


void Container::overwriteTree(int sourcePos, int targetPos){
       // replaces the tree targetPos with tree sourcePos
        if(sourcePos == targetPos){
            overwriteTree(targetPos);
        }else{
            delete this->trees[targetPos];
            this->trees[targetPos] = NULL;
            this->trees[targetPos] = new Tree(&this->nInstances, &this->nVariables, this->data, this->weights, this->trees[sourcePos]->splitV, this->trees[sourcePos]->splitP, this->trees[sourcePos]->csplit, &this->maxCat, &this->trees[sourcePos]->nNodes, this->variables, &this->maxNode);
            while( this->evaluateTree(targetPos, false, 0) == false){
                delete this->trees[targetPos];
                this->trees[targetPos] = NULL;
                int treeNo =  this->getRandomTree(true);
                while(treeNo == targetPos){
                        treeNo = this->getRandomTree(true);
                }
                this->trees[targetPos] = new Tree(&this->nInstances, &this->nVariables, this->data, this->weights, this->trees[treeNo]->splitV, this->trees[treeNo]->splitP, this->trees[treeNo]->csplit, &this->maxCat, &this->trees[treeNo]->nNodes, this->variables, &this->maxNode);
            }
        }
} // end overwriteTree



int Container::evaluateNewSolution(int treeNumber, double* oldPerf){
        // evaluates the improve of the new solution; is used by all mutation operators
        if(this->trees[treeNumber]->performance <= *oldPerf){
            return 1;
        }else
            return -1;
}  // end evaluateNewSolution


bool Container::updatePerformanceList(int treeNumber){
	// updates the performance list
            int newPos = -1;
            bool flag = true;
            for(int i = this->elitismRange-1; i >= 0 && flag == true ; i--){  
                if(this->elitismList[i] >= this->nTrees){ // list position is not filled with a treenumber
                    newPos = i;
                }else if(this->trees[treeNumber]->performance == this->trees[ this->elitismList[i]]->performance
                        && this->trees[treeNumber]->splitV[0] == this->trees[ this->elitismList[i]]->splitV[0]
                        && this->trees[treeNumber]->splitP[0] == this->trees[ this->elitismList[i]]->splitP[0]){
                        flag = false;  // likely the same tree is already in the elitism list (tree is not added)  
                }else if(this->trees[treeNumber]->performance < this->trees[this->elitismList[i] ]->performance){
                    newPos = i;
                }
            }
            if(newPos != -1 && flag == true ){
                for(int i = this->elitismRange-1; i > newPos; i--){
                    this->elitismList[i] = this->elitismList[i-1];
                }
                this->elitismList[newPos] = treeNumber;
            return true;
            }
        return false;
} // end updatePerformanceList


double Container::crossover(int treeNumber1){
        // variation operator crossover
        if( this->trees[treeNumber1]->nNodes < 3 ){ 
            return  this->initMutateNode(treeNumber1, true);
        }
	int treeNumber2 = this->getRandomTree(false);
        int i = 0;
	while( this->trees[treeNumber1]->performance ==  this->trees[treeNumber2]->performance || this->trees[treeNumber2]->nNodes < 2 ){
	 // search a second tree that have at least 2 internal nodes
            treeNumber2 = this->getRandomTree(false);
            i++;
            if (i == 50)
          	  return -2;
        }
        int randomNode1 = this->randomSplitNode(treeNumber1);
        i = 0;

	for(; i <= 30 &&  randomNode1 == 0; i++)
		randomNode1 = this->randomSplitNode(treeNumber1);
        if(i >= 30)
            return -2;
        int depthOfRandomNode = 0;
        bool flag = false;
        while(flag == false){
            if(randomNode1  >= pow(2,depthOfRandomNode+1)-1){  //smaller leftmost node of tree
                   depthOfRandomNode++;
            }else{
                   flag = true;
            }
        }
	int randomNode2 = (rand()%((int)pow(2,depthOfRandomNode+1) - (int)pow(2,depthOfRandomNode))) + (int)pow(2,depthOfRandomNode)-1;
        i = 0;
        while(this->trees[ treeNumber2 ]->splitV[randomNode2] < 0 && i <= 30){
            randomNode2 = (rand()%((int)pow(2,depthOfRandomNode+1)-(int)pow(2,depthOfRandomNode))) + (int)pow(2,depthOfRandomNode)-1;
            i++;
        }
        if(i >= 30)
            return -2; 
        int* splitV = new int[ this->maxNode ];
        double* splitP = new double[ this->maxNode];
        int **csplit;
        csplit = new int*[this->maxCat];
        int* splitV2 = new int[ this->maxNode];
        double* splitP2 = new double[ this->maxNode];
        int **csplit2;
        csplit2 = new int*[this->maxCat];
        // init data matrix
        for (int i = 0; i < this->maxCat; i++){
            csplit[i] = new int[(this->maxNode)];
            csplit2[i] = new int[(this->maxNode)];
        }

        for (int v = 0; v < this->maxNode ; v++){
            for(int i = 0; i < this->maxCat; i++){
                csplit[i][v] = 2;
                csplit2[i][v] = 2;
            }
            splitV[v] = -999999;
            splitP[v] = -999999;
            splitV2[v] = -999999;
            splitP2[v] = -999999;
        }

        int nNodes1 = initNVPCrossoverTree1(treeNumber1,           0, randomNode1, splitV, splitP, csplit) +
                   initNVPCrossoverTree2(treeNumber2, randomNode2, randomNode1, splitV, splitP, csplit);

        int nNodes2 = initNVPCrossoverTree1(treeNumber2,           0, randomNode2, splitV2, splitP2, csplit2) +
                   initNVPCrossoverTree2(treeNumber1, randomNode1, randomNode2, splitV2, splitP2, csplit2);

        Tree* oldTree = this->trees[treeNumber1];
        Tree* oldTree2 = this->trees[treeNumber2];
        this->trees[treeNumber1] = NULL;
        this->trees[treeNumber1] = new Tree(&this->nInstances, &this->nVariables, this->data, this->weights, splitV, splitP, csplit, &this->maxCat, &nNodes1, this->variables, &this->maxNode);
        this->trees[treeNumber2] = NULL;
        this->trees[treeNumber2] = new Tree(&this->nInstances, &this->nVariables, this->data, this->weights, splitV2, splitP2, csplit2, &this->maxCat, &nNodes2, this->variables, &this->maxNode);

       delete [] splitV;
       splitV = NULL;
       delete [] splitP;
       splitP = NULL;
       delete [] splitV2;
       splitV2 =NULL;
       delete [] splitP2;
       splitP2 = NULL;
       for (int i = 0; i < this->maxCat; i++){
            delete [] csplit[i];
            delete [] csplit2[i];
       }
       delete [] csplit;
       csplit = NULL;
       delete [] csplit2;
       csplit2 = NULL;

       int accept1 = 0;
       int accept2 = 0;

       if( this->evaluateTree(treeNumber1, true, 0) == false){
            accept1 = -1;
            // cout << "warning: invalid tree is replaced by a random tree (co 1)" << endl;
            this->overwriteTree(treeNumber1);
       }

       if( this->evaluateTree(treeNumber2, true, 0) == false){
            accept2 = -1;
            // cout << "warning: invalid tree is replaced by a random tree (co 2)" << endl;
            this->overwriteTree(treeNumber1);
       }

       if(accept2 == 0)
           accept2 = this->evaluateNewSolution(treeNumber2, &oldTree2->performance);
           if( accept2 >= 0){// new tree with "treeNumber2" is accepted
                delete oldTree2;
                oldTree2 = NULL;
            }else{
                delete this->trees[treeNumber2];
                this->trees[treeNumber2] = NULL;
                this->trees[treeNumber2] = oldTree2;
            }
      if(accept1 == 0)
            accept1 = this->evaluateNewSolution(treeNumber1, &oldTree->performance);
            if( accept1 >= 0){// new tree with "treeNumber1" is accepted
                     delete oldTree;
                     oldTree = NULL;
            }else{
                     delete this->trees[treeNumber1];
                     this->trees[treeNumber1] = NULL;
                     this->trees[treeNumber1] = oldTree;
            }

      return 1;  // improve
} // end crossover


int Container::initNVPCrossoverTree1(int treeNumber,int node, int randomNode, int* splitV, double* splitP, int** csplit){
     // counts the number of nodes in a subtree
     // used by crossover
      if( node < this->maxNode ){
          if( this->trees[treeNumber]->splitV[node] >= 0 && randomNode != node){
             splitV[node] = this->trees[treeNumber]->splitV[node];
             splitP[node] = this->trees[treeNumber]->splitP[node];
             for(int i = 0; i < this->maxCat; i++){
                csplit[i][node] = this->trees[treeNumber]->csplit[i][node];
             }
             return initNVPCrossoverTree1(treeNumber, node*2+1, randomNode, splitV, splitP, csplit) +
             initNVPCrossoverTree1(treeNumber, node*2+2, randomNode, splitV, splitP, csplit) + 1;
          }
      }
      return 0;
} // initNVPCrossoverTree1


int Container::initNVPCrossoverTree2(int treeNumber, int randomNode2, int randomNode1, int* splitV, double* splitP, int** csplit){
     // counts the number of nodes in a subtree
     // used by crossover
    if( randomNode1 < this->maxNode && randomNode2 < this->maxNode){
          if( this->trees[treeNumber]->splitV[randomNode2] >= 0){
             splitV[randomNode1] = this->trees[treeNumber]->splitV[randomNode2];
             splitP[randomNode1] = this->trees[treeNumber]->splitP[randomNode2];
             for(int i = 0; i < this->maxCat; i++){
                csplit[i][randomNode1] = this->trees[treeNumber]->csplit[i][randomNode2];
             }
             return initNVPCrossoverTree2(treeNumber, randomNode2*2+1, randomNode1*2+1, splitV, splitP, csplit) +
             initNVPCrossoverTree2(treeNumber, randomNode2*2+2, randomNode1*2+2, splitV, splitP, csplit) + 1;
          }
    }
    return 0;
} // initNVPCrossoverTree2
 
