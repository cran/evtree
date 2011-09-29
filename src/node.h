#include <iostream>
#include <R.h>
#include <Rinternals.h>
#include "variable.h"
#define pi 3.14159

using namespace std;

class Node{
	public:
	int pos;
	int* splitV;
	double* splitP;
        int** csplit;
	Node* leftChild;
	Node* rightChild;
	int* nInst;
	int* nVar;
	int* localClassification;
	double** data;
	int* nClassesDependendVar;
	variable **variables;

	double performanceLeftTerminal;
	double performanceRightTerminal;

	int sumLocalWeights; // weighted number of instances
	int sumLeftLocalWeights;
	int sumRightLocalWeights;

	double predictionInternalNode;
	double predictionLeftTerminal;
	double predictionRightTerminal;

	Node(int splitN, int* splitV, double* splitP, int** csplit, Node* leaftChild, Node* rightChild, double** data,
				int* nInst, int* nVar, variable** variables);
	~Node();
	int partition( int* classification, int* weights, variable** variables, int* nNodes, int minbucket, int minsplit);
	double calculateNodeSE(int* weights);
	double calculateNodeMC(int* weights);
        double calculateChildNodeMC(bool left, int* weights);
        double calculateChildNodeSE(bool left, int* weights);
        int factorial(int number);
};
