#include <iostream>
#include <Eigen/Dense>  
#include <math.h>
#include <fstream>

using namespace Eigen;
using namespace std;


int main()
{   
	MatrixXd Xtotal(2,48);
	Xtotal.col(0)<<0.25,0.75;

	MatrixXd z(1,49);
	
	MatrixXd P(2,2);
	P<<0,0,
	   0,0;
	 
	MatrixXd Eye2(2,2);
	Eye2<<1,0,
	      0,1;
	MatrixXd Hk(1,2);
	Hk<<1,1;
	      
	double sigmaH=0.1;
	double sigmaG=0.05;
	MatrixXd wk;
	MatrixXd Kk;
	MatrixXd Qk;
	MatrixXd Rk;
	
    double x;
    ifstream fileIN;
    fileIN.open("kfdata.txt");
    
    int colA=0;
	//Reading the Data file
	while(fileIN>>x) {
	z(0,colA)=x;
	colA++;
		}
	
	double theta=M_PI/24;
	MatrixXd Gk(2,2);
	Gk<<cos(theta),sin(theta),
	    -sin(theta),cos(theta);

	MatrixXd Sk;
	
	//Error Matrices:
	Qk=sigmaG*sigmaG*Eye2;
	Rk=sigmaH*sigmaH*MatrixXd::Identity(1,1);
	
	for(int k=0;k<48;k++)
	{
      if(k==0)
      {
		  Xtotal.col(k)=Gk*Xtotal.col(k);
	   }
      else
      {
		  Xtotal.col(k)=Gk*Xtotal.col(k-1);
	   }
   
	//Prediction for state vector and covariance:
	
	P=Gk*P*Gk.transpose()+Qk;
	
	//Measurement Residual and Residual covariance
	wk=z.col(k)-Hk*Xtotal.col(k);
	Sk=Hk*P*Hk.transpose()+Rk;
	
	//Compute the Kalman Gain
	Kk=P*Hk.transpose()*Sk.inverse();
	
	//Correction based on observation
	Xtotal.col(k)=Xtotal.col(k)+Kk*wk;
	P=P-Kk*Hk*P;
    }
    
    cout<<Xtotal<<endl;
   
	
	return 0;
	
}
