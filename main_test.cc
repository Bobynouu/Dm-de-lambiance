#include "DM_test.h"  // Inclure le fichier .h

using namespace std;
using namespace Eigen;

int main()
{
  int userChoiceMeth(0), nom_matrice(0);
  int n_ite_max(20000000);
  double eps(0.01);
  int  N(1);
  double alpha(3*N);
  string name_file;
  SparseMatrix<double> In(N,N) , Bn(N,N) , An(N,N), BnTBn(N,N);
  VectorXd b(N), x0(N);

//Choix de la matrice à utiliser.
  cout<<"Quelle matrice souhaitez-vous ?"<<endl;
 cout << "---------------------------------" << endl;
 cout << "1) s3rmt3m3 (N=5357)" <<endl;
 cout << "2) bcsstm24 (N=3562)" <<endl;
 cout << "3) matrice de la forme alpha.In + Bnt.Bn (commenter la ligne create_mat dans les case des methode résolution)" <<endl;
 // cout << "fs_541_4 (4)" <<endl;
 // cout << "cas général (5)"<< endl;
 cin >> nom_matrice ;

 MethIterative *Solv(0);

 switch(nom_matrice)
 {
   case 1:
    name_file = "s3rmt3m3.mtx" ;
        cout << "N= ?"<< endl;
      cin >> N;
      x0.resize(N) , b.resize(N);
   break;

   case 2:
        name_file = "bcsstm24.mtx";
        cout << "N= ?"<< endl;
        cin >> N;
        x0.resize(N) , b.resize(N);
   break;

     default:

     cout << "3) matrice de la forme alpha.In + Bnt.Bn" <<endl;
        cout << "N= ?"<< endl;
        cin >> N;

     In.resize(N,N), Bn.resize(N,N) , An.resize(N,N), BnTBn.resize(N,N) , x0.resize(N) , b.resize(N);
     In.setIdentity();
     //création de Bn

     for (int i =0 ; i<N ; i++)
          {
              for (int j =0 ; j<N ; j++)
                  {double nb_alea = rand()/(double)RAND_MAX  ;
                   Bn.coeffRef(i,j) = nb_alea;  }
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



        for( int i = 0 ; i < N ; ++i)
          {x0(i) = 1.;}

        // création de b
         b = An*x0;
        cout <<"    "<<endl;
        cout <<"b = "<<endl<< b <<endl;
        cout <<"    "<<endl;
        cout <<"b est créé"<<endl;


        for( int i = 0 ; i < N ; ++i)
          {x0.coeffRef(i) = 2.;}
            cout <<"x0 est créé"<<endl;

     exit(0);
   }

  cout << "Veuillez choisir la méthode de résolution pour Ax=b:" << endl;
  cout << "1) Méthode du Résidu Minimum" << endl;
  cout << "2) Méthode du Gradient Conjugué" << endl;
  cout << "3) Méthode SGS" << endl;
  cin >> userChoiceMeth;




//  SparseMatrix<double> In(N,N) , Bn(N,N) , An(N,N), BnTBn(N,N);
  VectorXd x(N) ,x1(N) ,x2(N);

  x.setZero();
  x1.setZero();
  x2.setZero();

  int n_ite(0);
  VectorXd z(N);

  // on construit le residu minimum
  MethIterative* MethIterate(0);

  ofstream mon_flux; // Contruit un objet "ofstream"


  switch(userChoiceMeth)
  {
    case 1:
      MethIterate = new ResiduMin();
      An = MethIterate->create_mat(name_file, true );
      cout<<"An créé"<<endl;
      N = An.rows();

      for( int i = 0 ; i < N ; ++i)                                     // b est créé pour que la solution exacte soit x=(1,1,....,1)
        {x0(i) = 1.;}
      b = An*x0;
      cout<<"b créé"<<endl;

      for( int i = 0 ; i < N ; ++i)
          {x0.coeffRef(i) = 2.;}
      cout <<"x0 est créé"<<endl;


      MethIterate->MatrixInitialize(An);
      MethIterate->Initialize(x0, b);

      name_file = "soltest2"+to_string(N)+"_res_min.txt";
      mon_flux.open(name_file);

      while(MethIterate->GetResidu().norm() > eps && n_ite < n_ite_max)
      {
      //  cout<<"début while"<<endl;
        z = An*MethIterate->GetResidu();
      //  cout<<"milieu while"<<endl;
        MethIterate->Advance(z);

        n_ite++;
        if(mon_flux)
          {
              mon_flux<<n_ite<<" "<<MethIterate->GetResidu().norm()<<endl;
          }
      }

      if(n_ite >= n_ite_max)
        {cout << "Tolérance non atteinte"<<endl;}

      cout <<"  "<<endl;   // d'après doc internet si x0 est trop éloigné de x les résultats ne converge plus
    //  cout <<"x avec res_min = "<<endl << MethIterate->GetIterateSolution() <<endl;
      cout <<"    "<<endl;
    //  cout << An*MethIterate->GetIterateSolution()<<endl;
      cout <<"    "<<endl;
      cout << "nb d'itérations pour res min = " << n_ite << endl;

      break;

    case 2:
      MethIterate = new GradientConj();

      cout<<"début création de An"<<endl;
      An = MethIterate->create_mat(name_file, true );
      cout<<"An créé"<<endl;


      for( int i = 0 ; i < N ; ++i)           // b est créé pour que la solution exacte soit x=(1,1,....,1)
        {x0(i) = 1.;}
      b = An*x0;
      cout<<"b créé"<<endl;

      for( int i = 0 ; i < N ; ++i)
          {x0.coeffRef(i) = 2.;}
      cout <<"x0 est créé"<<endl;

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

      if (n_ite >= n_ite_max)
        {cout << "Tolérance non atteinte"<<endl;}

      cout <<"  "<<endl;   // d'après doc internet si x0 est trop éloigné de x les résultats ne converge plus
    //  cout <<"x avec grad_conj = "<<endl << MethIterate->GetIterateSolution() <<endl;
      cout <<"    "<<endl;
    //  cout << An*MethIterate->GetIterateSolution()<<endl;
      cout <<"    "<<endl;
      cout << "nb d'itérations pour grad conj = " << n_ite << endl;

      MethIterate->Get_norme_sol();

      break;

    case 3:
      MethIterate = new SGS();

      An = MethIterate->create_mat(name_file, true );
      cout<<"An créée"<<endl;
      for( int i = 0 ; i < N ; ++i)           // b est créé pour que la solution exacte soit x=(1,1,....,1)
        {x0(i) = 1.;}
      b = An*x0;
      cout<<"b créé"<<endl;

      for( int i = 0 ; i < N ; ++i)
          {x0.coeffRef(i) = 2.;}
      cout <<"x0 est créé"<<endl;

      MethIterate->MatrixInitialize(An);
      MethIterate->Initialize(x0, b);


      name_file = "sol"+to_string(N)+"_SGS.txt";
      mon_flux.open(name_file);

      while(MethIterate->GetResidu().norm() > eps && n_ite < n_ite_max)
      {
        z = An*MethIterate->GetIterateSolution();
        MethIterate->Advance(z);
        cout<<n_ite<<endl;
        n_ite++;
          if(mon_flux)
            {
              mon_flux<<n_ite<<" "<<MethIterate->GetResidu().norm()<<endl;
            }
      }


      if (n_ite >= n_ite_max)
        {cout << "Tolérance non atteinte"<<endl;}

      cout <<"  "<<endl;   // d'après doc internet si x0 est trop éloigné de x les résultats ne converge plus
      cout <<"x avec SGS = "<<endl << MethIterate->GetIterateSolution() <<endl;
      cout <<"    "<<endl;
      cout << An*MethIterate->GetIterateSolution()<<endl;
      cout <<"    "<<endl;
      cout << "nb d'itérations pour SGS = " << n_ite << endl;
      break;



    default:
      cout << "Ce choix n'est pas possible" << endl;
      exit(0);
  }


  //CALCUL NORME THEORIQUE


//  cout<<"norm(An) "<<An.norm()<<endl;

  delete MethIterate;
  MethIterate = 0;
  mon_flux.close();




  return 0;
}
