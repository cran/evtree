#include "node.h"
using namespace std;

class Tree{
		public:
	     	int *nInstances;
	 		int *nVariables;
		variable **variables;
			double **data;
		int *maxNode;
			 int * maxCat;
		 int *splitV;
		 double *splitP;
			 int *weights;
			 int** csplit;
		 int nNodes;
		 int *classification;	
		 int *posDependendVar;
		 Node **nodes;
		 double performance;  

	public:
		Tree(int* nInst, int* nVariables, double** data, int* weights, int *splitV, double *splitP, int** csplit, int* maxCat, int* nNodes, variable** variables, int* maxNode);
		Tree(int* nInst, int* nVariables, double** data, int* weights, int* maxCat, variable** variables, int* maxNode, int* minBucket, int* minSplit);
		~Tree();
		int predictClass(int minBucket, int minSplit, bool crossover, int nodeNumber);
		void initNode(int nodeNo);
		double calculateTotalMC(int nodeNumber);
		bool calculateTotalCosts(int method, double alpha, int sumWeights, double populationMSE);
		double calculateTotalSE(int nodeNumber);
		bool deleteChildNodes(int nodeNo);
		bool reverseClassification(int startNode, int nodeNumber);
		void randomizeCategories(int nodeNumber);
		int getRandomNumberIntBetween(int min_value, int max_value);
		static int getUnifRandNumber(int numberDistinctValues);		
};
