#include "DM_test.h"  // Inclure le fichier .h

using namespace std;
using namespace Eigen;

int main()
{
  int userChoiceMeth(0);
  int n_ite_max(200000);
  double eps(0.01);
  int const N(5);
  double alpha(3*N);
  string name_file;

  cout << "Veuillez choisir la méthode de résolution pour Ax=b:" << endl;
  cout << "1) Méthode du Résidu Minimum" << endl;
  cout << "2) Méthode du Gradient Conjugué" << endl;
  cout << "3) Méthode SGS" << endl;
  cout << "4) Méthode Gmres" << endl;
  cout << "5) Méthode Gradient conjugué préconditionné" << endl;
  cin >> userChoiceMeth;

//création d'une matrice de la forme souhaitée
SparseMatrix<double> In(N,N) , Bn(N,N) , An(N,N), BnTBn(N,N);
VectorXd b(N) ,x(N) ,x1(N) ,x2(N);

x.setZero();
x1.setZero();
x2.setZero();

In.setIdentity();

//création de Bn

for (int i =0 ; i<N ; i++)
  {
    for (int j =0 ; j<N ; j++)
      {
      double nb_alea = rand()/(double)RAND_MAX  ;
      Bn.coeffRef(i,j) = nb_alea;
    }
  }


//Création de An
BnTBn = Bn.transpose()*Bn;
for (int i =0 ; i< N; i++)
  {
    for (int j =0 ; j< N; j++)
    {
      An.insert(i,j) =alpha*In.coeffRef(i,j) + BnTBn.coeffRef(i,j);
    }
  }

 //cout <<"An = "<<endl<< An <<endl;

 //création de b
 for (int i =0 ; i< N;i++)
  {b.coeffRef(i) = 1. + i;}
// cout <<"    "<<endl;
// cout <<"b = "<<endl<< b <<endl;
// cout <<"    "<<endl;


  // creation de x0
  VectorXd x0(N);
  for( int i = 0 ; i < N ; ++i)
  {
    x0.coeffRef(i)=1.;
  }

  int n_ite(0);
  VectorXd z(N);

  // on construit le residu minimum
  MethIterative* MethIterate(0);
  Gmres gmrs;
  GradientConPrecond gradprecond;

  ofstream mon_flux; // Contruit un objet "ofstream"

  // pour gradient conjugué preconditionné
  SparseLU <SparseMatrix<double>> solver_xnew;
  VectorXd X_Sol;

  switch(userChoiceMeth)
  {
    case 1:
      MethIterate = new ResiduMin();
      MethIterate->MatrixInitialize(An);
      MethIterate->Initialize(x0, b);

      name_file = "sol"+to_string(N)+"_res_min.txt";
      mon_flux.open(name_file);

      while(MethIterate->GetResidu().norm() > eps && n_ite < n_ite_max)
      {
        z = An*MethIterate->GetResidu();
        MethIterate->Advance(z);
        n_ite++;
        if(mon_flux)
          {
              mon_flux<<n_ite<<" "<<MethIterate->GetResidu().norm()<<endl;
          }
      }

      if(n_ite > n_ite_max)
        {cout << "Tolérance non atteinte"<<endl;}

      cout <<"  "<<endl;   // d'après doc internet si x0 est trop éloigné de x les résultats ne converge plus
      cout <<"x avec res_min = "<<endl << MethIterate->GetIterateSolution() <<endl;
      cout <<"    "<<endl;
      cout << An*MethIterate->GetIterateSolution()<<endl;
      cout <<"    "<<endl;
      cout << "nb d'itérations pour res min = " << n_ite << endl;

      break;

    case 2:
      MethIterate = new GradientConj();
      MethIterate->MatrixInitialize(An);
      MethIterate->Initialize(x0, b);

      name_file = "sol"+to_string(N)+"_grad_conj.txt";
      mon_flux.open(name_file);

      while(MethIterate->GetResidu().norm() > eps && n_ite < n_ite_max)
      {
        z = An*MethIterate->Getp();
        MethIterate->Advance(z);
        n_ite++;
        if(mon_flux)
          {
              mon_flux<<n_ite<<" "<<MethIterate->GetResidu().norm()<<endl;
          }
      }

      if (n_ite > n_ite_max)
        {cout << "Tolérance non atteinte"<<endl;}

      cout <<"  "<<endl;   // d'après doc internet si x0 est trop éloigné de x les résultats ne converge plus
      cout <<"x avec grad_conj = "<<endl << MethIterate->GetIterateSolution() <<endl;
      cout <<"    "<<endl;
      cout << An*MethIterate->GetIterateSolution()<<endl;
      cout <<"    "<<endl;
      cout << "nb d'itérations pour grad conj = " << n_ite << endl;

      break;

    case 3:
      MethIterate = new SGS();
      MethIterate->MatrixInitialize(An);
      MethIterate->Initialize(x0, b);


      name_file = "sol"+to_string(N)+"_SGS.txt";
      mon_flux.open(name_file);

      while(MethIterate->GetResidu().norm() > eps && n_ite < n_ite_max)
      {
        z = An*MethIterate->GetIterateSolution();
        MethIterate->Advance(z);
        n_ite++;
          if(mon_flux)
            {
              mon_flux<<n_ite<<" "<<MethIterate->GetResidu().norm()<<endl;
            }

      }


      if (n_ite > n_ite_max)
        {cout << "Tolérance non atteinte"<<endl;}

      cout <<"  "<<endl;   // d'après doc internet si x0 est trop éloigné de x les résultats ne converge plus
      cout <<"x avec SGS = "<<endl << MethIterate->GetIterateSolution() <<endl;
      cout <<"    "<<endl;
      cout << An*MethIterate->GetIterateSolution()<<endl;
      cout <<"    "<<endl;
      cout << "nb d'itérations pour SGS = " << n_ite << endl;
      break;


    case 4:
      gmrs.MatrixInitialize(An);
      gmrs.Initialize(x0, b);

      name_file = "sol"+to_string(N)+"_GMRes.txt";
      mon_flux.open(name_file);

      while(gmrs.GetNorm() > eps && n_ite < n_ite_max)
      {
        // cout << "An = " << An <<endl;
        // cout << "residu = " <<gmrs.GetResidu()<<endl;
        gmrs.Arnoldi(An, gmrs.GetResidu()); // check le remplissage de la SparseMatrix
        //cout << "HM " << gmrs.GetHm() << endl;
        gmrs.Givens(gmrs.GetHm()); // attention renvoi Rm et Qm de taille carré
        gmrs.Advance(gmrs.GetResidu());

        n_ite++;
      }
      if (n_ite > n_ite_max)
        {cout << "Tolérance non atteinte"<<endl;}

      cout <<"  "<<endl;   // d'après doc internet si x0 est trop éloigné de x les résultats ne converge plus
      cout <<"x avec SGS = "<<endl << gmrs.GetIterateSolution() <<endl;
      cout <<"    "<<endl;
      cout << An*gmrs.GetIterateSolution()<<endl;
      cout <<"    "<<endl;
      cout << "nb d'itérations pour SGS = " << n_ite << endl;
      break;

      case 5:

        gradprecond.MatrixInitialize(An);
        gradprecond.Initialize(x0,b);

        name_file = "sol"+to_string(N)+"_grad_conj_precond.txt";
        mon_flux.open(name_file);

        while(gradprecond.GetResidu().norm() > eps && n_ite < n_ite_max)
        {
          VectorXd U, V, W;
          U.resize(An.cols()),V.resize(An.cols()),W.resize(An.cols());
          SparseLU <SparseMatrix<double>> solver_u, solver_z;

          solver_u.compute(gradprecond.Get_M());
          U = solver_u.solve(gradprecond.Getp());
          V = An*U;
          W = An.transpose()*V;
          solver_z.compute(gradprecond.Get_M().transpose());
          z = solver_z.solve(W);

          gradprecond.Advance(z);
          n_ite++;
          if(mon_flux)
            {
                mon_flux<<n_ite<<" "<<gradprecond.GetResidu().norm()<<endl;
            }
        }

        if (n_ite > n_ite_max)
          {cout << "Tolérance non atteinte"<<endl;}
// on a dans get iterate solution y = Mx il faut resoudre pour avoir x
        X_Sol.resize(An.cols());
        solver_xnew.compute(gradprecond.Get_M());
        X_Sol = solver_xnew.solve(gradprecond.GetIterateSolution());
        
        cout <<"  "<<endl;   // d'après doc internet si x0 est trop éloigné de x les résultats ne converge plus
        cout <<"x avec grad_conj_precond = "<<endl << X_Sol <<endl;
        cout <<"    "<<endl;
        cout << "devrait valoir b :";
        cout << An*X_Sol<<endl;
        cout <<"    "<<endl;
        cout << "nb d'itérations pour grad conj preconditionné = " << n_ite << endl;

        break;


    default:
      cout << "Ce choix n'est pas possible" << endl;
      exit(0);
  }


  delete MethIterate;
  MethIterate = 0;
  mon_flux.close();


  return 0;
}
