#include <iostream>
#include <math.h>
#include <R.h>
#include <Rinternals.h>
#include <R_ext/Utils.h>
#include <cstdlib>
#include <ctime> 
#include "tree.h"

using namespace std;

class Container{
	 public:
         int nInstances;
         int nVariables;
	 variable **variables;
         double **data;
         int *weights;
         int* elitismList;
         int nTrees;
         int minbucket;
         int minsplit;
         int maxNode; // maximum number of nodes a tree with maxdepth level can have
         int maxCat;  // maximum number of categories of nominal variables
         int nIterations;
         double probMutateMajor;
         double probMutateMinor;
         double probSplit;
         double probPrune;
         double probCrossover;
         double *performanceHistory; // average performance of the trees in the elitism list, is used for the termination of the algorithm
         Tree **trees;
         int elitismRange; // number of trees in the elitism list
         int method; // method=1 for classification, and method=6 for regression / other criteria are not implemented at the moment
         double alpha;  // weights the complexity part of the cost function
         int sumWeights; //  sum of weights 
         double populationMSE; // population variance; used for regression trees only
         public:
         Container(int* nInst, int* nVar, int *varType, double* ndata, int* weights, int* prediction, int *splitV, double *splitP, int* csplit, int *maxNode, int *minbucket, int* minsplit,
              int* nIter, int* nTrees, int* pMutateMajor, int* pMutateMinor, int* pCrossover, int *pSplit, int* pPrune, int* evCriteria, double* evParameter, int* seed);
         ~Container();
         void initVariables(int* varType);
         bool evolution();
         bool checkInterrupt(void);
         bool evaluateTree(int treeNumber, bool pruneIfInvalid, int nodeNumber);
         double initMutateNode(int treeNumber, bool minorChange);
         double mutateNode(int treeNumber, int node, bool minorChange);
         int calculateNoOfNodesInSubtree(int treeNumber, int nodeNumber);
         double splitNode(int treeNumber);
         double pruneNode(int treeNumber);
         int pruneAllNodes(int treeNumber);
         double crossover(int treeNumber);
         int getRandomTree(bool elitismTree);
         int getGenitor(void);
         int randomSplitNode(int treeNumber);
         int randomTerminalNode(int treeNumber);
         void randomSplitVariable(int treeNumber, int nodeNumber);
         bool randomSplitPoint(int treeNumber, int nodeNumber);
         bool changeSplitPoint(int treeNumber, int nodeNumber);
         bool changeRandomCategories(int treeNumber, int nodeNumber);
         void overwriteTree(int targetPos);
         void overwriteTree(int sourcePos, int targetPos);
         int evaluateNewSolution(int treeNumber, double* oldPerformance);
         bool updatePerformanceList(int tree);
         int initNVPCrossoverTree1(int treeNumber, int node, int randomNode1, int* tempV, double* tempP, int** csplit);
         int initNVPCrossoverTree2(int treeNumber, int randomNode2, int randomNode1, int* tempV, double* tempP, int** csplit);
};

