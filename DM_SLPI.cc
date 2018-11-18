
#include "DM_SLPI.h"  // Inclure le fichier .h
#include <vector>
#include <iostream>
#include <string>
#include <cmath>
#include <fstream>
#include <sstream>
#include <chrono>
#include<complex>
#include <cstdlib>
#include <ctime>
#include <random>

#include "Dense"
#include "Sparse"

using namespace std;
using namespace Eigen;

int main()
{
//création d'une matrice de la forme souhaitée
int const n(1000.);                                          //taille de la matrice
double alpha(2.);
SparseMatrix<double> In(n,n) , Bn(n,n) , An(n,n), BnTBn(n,n);
SparseVector<double> b(n) ,x(n) ,x1(n) ,x2(n);

x.setZero();
x1.setZero();
x2.setZero();

In.setIdentity();

//création de Bn
const int min=0;
const int max=2;
for (int i =0 ; i<n ; i++)
  {
    for (int j =0 ; j<n ; j++)
      {
      double nb_alea = min + (rand()%(max-min));
      Bn.coeffRef(i,j) = nb_alea;
    }
  }
//  cout <<"Bn"<<Bn<<endl;

//Création de An
BnTBn = Bn.transpose()*Bn;
for (int i =0 ; i<n ; i++)
  {
    for (int j =0 ; j<n ; j++)
      {
        An.insert(i,j) =alpha*In.coeffRef(i,j) + BnTBn.coeffRef(i,j);

}
}
 //cout <<"An = "<<endl<< An <<endl;

 //création de b
 for (int i =0 ; i<n ;i++)
  {b.coeffRef(i) = 1. +i;}
// cout <<"    "<<endl;
// cout <<"b = "<<endl<< b <<endl;
// cout <<"    "<<endl;

//RESOLUTION

//SGS/////////////////////////////////////////////////////////////////////////////////////////////
// cout <<"  "<<endl;
// cout<<"Avec Gauss Seidel symétrique"<<endl;
// cout <<"  "<<endl;
// x = SGS( An , b ,  x , 0.01 , 20000 );
// cout <<"x avec SGS = "<<endl << x <<endl;
// cout <<"    "<<endl;
// cout <<"Vérification en calculant An.x = "<<endl<< An*x<<endl;

//RESIDU MINIMUM//////////////////////////////////////////////////////////////////////////////////////////

cout<<"---------------------------------------"<<endl;
cout<<"Avec res min"<<endl;
cout <<"    "<<endl;
SparseVector<double> x0;
x0.resize(x.size());
for( int i = 0 ; i < x.size() ; ++i)
  {x0.coeffRef(i) = 0.;}

 x1 = res_min( An , b ,  x1 , x0 , 0.01 , 200000 );   //ne fonctionne pas si l'on met autre chose que 0 dans x0
// cout <<"  "<<endl;                                // d'après doc internet si x0 est trop éloigné de x les résultats ne converge plus
// cout <<"x avec res_min = "<<endl << x1 <<endl;
// cout <<"    "<<endl;
// cout << "A*x = "<<endl<<An*x1<<endl;
// cout <<"    "<<endl;




// //GRADIENT CONJUGUE//////////////////////////////////////////////////////////////////////////////
// cout<<"---------------------------------------"<<endl;
// cout<<"Avec le gradient conjugué"<<endl;
// cout <<"    "<<endl;
//
//
 x2 = grad_conj( An , b , x ,  x0,  0.01 , 200000 );  //ne fonctionne pas si l'on met autre chose que 0 dans x0
// cout <<"  "<<endl;
// cout <<"x avec grad_conj = "<<endl << x2 <<endl;
// cout <<"    "<<endl;
// cout << "Vérification en calculant An*x = "<< endl << An*x2 <<endl;


cout<<"---------------------------------------"<<endl;
cout<<"Avec le gradient conjugué"<<endl;
cout <<"    "<<endl;
//bool sym(TRUE);
//SparseMatrix<double> S3 = create_mat("s3rmt3m3.mtx", true );

//x2 = grad_conj( An , b , x ,  x0,  0.1 , 20000 );  //ne fonctionne pas si l'on met autre chose que 0 dans x0
// cout <<"  "<<endl;
// cout <<"x avec grad_conj = "<<endl << x2 <<endl;
// cout <<"    "<<endl;
// cout << "Vérification en calculant An*x = "<< endl << An*x2 <<endl;
//EN creux
// SparseMatrix<double> Sn ;
// cout <<"    "<<endl;
// cout <<"En creux = "<<endl;
// cout <<"    "<<endl;
// Sn = create_mat("test.mtx", false);
// cout <<"Sn = "<<Sn<<endl;

return 0;
}
