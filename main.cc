#include "Solve_lin.h"  // Inclure le fichier .h

using namespace std;
using namespace Eigen;

int main()
{
  // Initialisation des variables
  int userChoiceMeth(0), MatrixChoice(0.);
  int n_ite_max(200000), n_ite(0);
  double eps(0.01);
  int N(0);
  double alpha(3.);
  SparseMatrix<double> A;
  VectorXd b, x0;
  bool sym;          // indique si la matrice est symétrique(true) ou non(false)
  string name_file;  // nom du fichier contenant la norme des résidus
  SparseMatrix<double> In, Bn, BnTBn;

  //Choix de la matrice à utiliser.
  cout<<"Quelle matrice souhaitez-vous ?"<<endl;
  cout << "---------------------------------" << endl;
  cout << "1) s3rmt3m3 (N=5 357)" <<endl;
  cout << "2) bcsstm24 (N=3 562)" <<endl;
  cout << "3) af23560 (N=23 560)" << endl;
  cout << "4) MCCA (N=180)" << endl;
  cout << "5) matrice de la forme alpha.In + Bnt.Bn (commenter la ligne create_mat dans les case des methode résolution)" << endl;
  cin >> MatrixChoice ;

  // choix de la matrice selon le choix de l'utilisateur
  switch(MatrixChoice)
  {
    case 1:
      name_file = "s3rmt3m3.mtx" ;
      sym = true;
      N = 5357;
    break;

    case 2:
      name_file = "bcsstm24.mtx";
      sym = true;
      N = 3562;
    break;

    case 3:
      name_file = "af23560.mtx";
      sym = true;
      N = 23560;
    break;

    case 4:
      name_file = "MCCA.mtx";
      sym = false;
      N = 180;
    break;

    case 5:
      cout << "Donnez moi la dimension de la matrice de la forme alphaIn + Bn*Bn.transpose" << endl;
      cin >> N;
      sym = true;
      alpha = 3*N;   // Afin d"avoir une matrice Ultra bien conditionnée.

      // initialisation de la construction de An
      In.resize(N,N), Bn.resize(N,N), BnTBn.resize(N,N);
      In.setIdentity();
      A.resize(N,N);

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
          A.insert(i,j) =alpha*In.coeffRef(i,j) + BnTBn.coeffRef(i,j);
        }
      }
      break;

    default:
      cout << "Ce choix n'est pas possible" << endl;
      exit(0);
  }

  // Méthode utilisée selon le choix de l'utilisateur
  cout << "Veuillez choisir la méthode de résolution pour Ax=b:" << endl;
  cout << "1) Méthode du Résidu Minimum" << endl;
  cout << "2) Méthode du Gradient Conjugué" << endl;
  cout << "3) Méthode SGS" << endl;
  cout << "4) Méthode GMRes" << endl;
  cin >> userChoiceMeth;


  //création de b
  b.resize(N);
  for (int i=0; i< N; i++)
    {b.coeffRef(i) = 1. + i;}

  // creation de x0
  x0.resize(N);
  for( int i = 0 ; i < N ; ++i)
  {
    x0.coeffRef(i)=2.;
  }

  // on construit l'objet de la méthode choisie
  MethIterative* MethIterate(0);  // si ce n'est pas GMRes
  Gmres gmrs;                     // pour la méthode de GMRes

  // Contruit un objet "ofstream" afin d'écrire dans un fichier
  ofstream mon_flux;

  // Algorithme de la méthode itérative choisie
  VectorXd z(N);           // Utile à la méthode pour éviter les calculs superflus

  switch(userChoiceMeth)
  {
  ///////////////// Résidu Minimum /////////////////////
    case 1:
      // Initialisation
      MethIterate = new ResiduMin();
      if (MatrixChoice != 5)    // formation de la matrice dans le cas des matrices venant de MatrixMarket
        {A = MethIterate->create_mat(name_file, sym);}
      MethIterate->MatrixInitialize(A);
      MethIterate->Initialize(x0, b);
      name_file = "sol"+to_string(N)+"_res_min.txt";
      mon_flux.open(name_file);

      // Corps du programme
      while(MethIterate->GetResidu().norm() > eps && n_ite < n_ite_max)
      {
        z = A*MethIterate->GetResidu();
        MethIterate->Advance(z);
        n_ite++;
        // On sauvgarde la norme du résidu dans un fichier
        if(mon_flux)
        {
          mon_flux<<n_ite<<" "<<MethIterate->GetResidu().norm()<<endl;
        }
      }

      if(n_ite > n_ite_max)
        {cout << "Tolérance non atteinte" << endl;}

      // Vérification de la solution obtenue
      cout <<"  "<<endl;
      cout<<"norme de b-Ax = "<<Verif_norme(MethIterate->GetIterateSolution() , b , A)<<endl;
      cout <<"    "<<endl;
      cout << "nb d'itérations pour res min = " << n_ite << endl;
      break;

///////////////// Grandient Conjugué /////////////////////
    case 2:
      // Initialisation
      MethIterate = new GradientConj();
      if (MatrixChoice != 5)     // formation de la matrice dans le cas des matrices venant de MatrixMarket
        {A = MethIterate->create_mat(name_file, sym);}
      MethIterate->MatrixInitialize(A);
      MethIterate->Initialize(x0, b);
      name_file = "sol"+to_string(N)+"_grad_conj.txt";
      mon_flux.open(name_file);

      // Corps du programme
      while(MethIterate->GetResidu().norm() > eps && n_ite < n_ite_max)
      {
        z = A*MethIterate->Getp();
        MethIterate->Advance(z);
        n_ite++;
        // On sauvgarde la norme du résidu dans un fichier
        if(mon_flux)
        {
          mon_flux<<n_ite<<" "<<MethIterate->GetResidu().norm()<<endl;
        }
      }

      if (n_ite > n_ite_max)
        {cout << "Tolérance non atteinte"<<endl;}

        // Vérification de la solution obtenue
        cout <<"  "<<endl;
        cout<<"norme de b-Ax = "<<Verif_norme(MethIterate->GetIterateSolution() , b , A)<<endl;
        cout <<"    "<<endl;
        cout << "nb d'itérations pour res min = " << n_ite << endl;
      break;

///////////////// Gauss Seildel Symétrique /////////////////////
    case 3:
      // Initialisation
      MethIterate = new SGS();
      if (MatrixChoice != 5)      // formation de la matrice dans le cas des matrices venant de MatrixMarket
        {A = MethIterate->create_mat(name_file, sym);}
      MethIterate->MatrixInitialize(A);
      MethIterate->Initialize(x0, b);
      name_file = "sol"+to_string(N)+"_SGS.txt";
      mon_flux.open(name_file);

      // Corps du programme
      while(MethIterate->GetResidu().norm() > eps && n_ite < n_ite_max)
      {
        z = A*MethIterate->GetIterateSolution();
        MethIterate->Advance(z);
        n_ite++;
        // On sauvgarde la norme du résidu dans un fichier
        if(mon_flux)
        {
          mon_flux<<n_ite<<" "<<MethIterate->GetResidu().norm()<<endl;
        }
      }

      if (n_ite > n_ite_max)
        {cout << "Tolérance non atteinte"<<endl;}

        // Vérification de la solution obtenue
        cout <<"  "<<endl;
        cout<<"norme de b-Ax = "<<Verif_norme(MethIterate->GetIterateSolution() , b , A)<<endl;
        cout <<"    "<<endl;
        cout << "nb d'itérations pour res min = " << n_ite << endl;
      break;

///////////////// GMRes /////////////////////
    case 4:
      // Initialisation
      if (MatrixChoice != 5)   // formation de la matrice dans le cas des matrices venant de MatrixMarket
        {A = MethIterate->create_mat(name_file, sym);}

      gmrs.MatrixInitialize(A);

      gmrs.Initialize(x0, b);
      name_file = "sol"+to_string(N)+"_GMRes.txt";
      mon_flux.open(name_file);

      // Corps du programme
      while(gmrs.GetNorm() > eps && n_ite < n_ite_max)
      {
        gmrs.Arnoldi(A, gmrs.GetResidu()); // on fabrique Hm à partir de A et du résidu

        gmrs.Givens(gmrs.GetHm());    // On fait la décomposition QR de Hm adaptée aux matrices d'Hessenberg

        gmrs.Advance(gmrs.GetResidu());  // Itération
        n_ite++;
      }
      if (n_ite > n_ite_max)
        {cout << "Tolérance non atteinte"<<endl;}

        // Vérification de la solution obtenue
        cout<<"norme de b-Ax = "<<Verif_norme(gmrs.GetIterateSolution() , b , A)<<endl;
        cout <<"    "<<endl;
        cout << "nb d'itérations pour res min = " << n_ite << endl;
      break;

    default:
      cout << "Ce choix n'est pas possible" << endl;
      exit(0);
  }

  // Libération de la mémoire
  delete MethIterate;
  MethIterate = 0;
  mon_flux.close();

  return 0;
}