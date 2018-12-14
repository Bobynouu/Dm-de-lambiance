#ifndef _Meth_Iterative_H
#include "Solve_lin.h"  // Inclure le fichier .h

using namespace std;
using namespace Eigen;

// constructeur par défaut
MethIterative::MethIterative()
{}

// Destructeur
MethIterative::~MethIterative()
{}

/////////////////////////////////////////////////////////
///////////////////// CLASSE MERE ///////////////////////
/////////////////////////////////////////////////////////
void MethIterative::MatrixInitialize(SparseMatrix<double> A)
{
  _A.resize(A.rows(), A.cols());
  _A = A;   // A prend la valeur de la matrice du système
}
/////////////////////////////////////////////////////////
const VectorXd & MethIterative::GetIterateSolution() const
{
  return _x;  // On récupère la solution itérée
}
/////////////////////////////////////////////////////////
const VectorXd & MethIterative::GetResidu() const
{
  return _r;  // On récupère le résidu
}
/////////////////////////////////////////////////////////
const VectorXd & MethIterative::Getp() const
{
  return _p;  // On récupère le vecteur p
}
/////////////////////////////////////////////////////////
void MethIterative::saveSolution(int N ,string name_file , int n_iter , double residu)
{
  ofstream mon_flux; // Contruit un objet "ofstream"
  // name_file = ("/sol_"+to_string(N)+"_"+name_file+".txt");  //commande pour modifier le nom de chaque fichier
  mon_flux.open(name_file,ios::out);

  if(mon_flux)
  {
    mon_flux<<n_iter<<residu<<endl;
  }
  mon_flux.close();
}
/////////////////////////////////////////////////////////
SparseMatrix<double>  MethIterative::create_mat(const string name_file_read, bool sym)
  {
    int N(0.);
    SparseMatrix<double> A ;
    ifstream mon_flux(name_file_read);
    string ligne, colonne, valeur;
    getline(mon_flux,ligne); //lit la première ligne qui ne nous intéresse pas

    mon_flux >> N; //lit le premier mot de la ligne 2 correspond au nombre de lignes
    int nonzero_elem;
    mon_flux >> colonne; //lit le nombre de colonnes (valeur stockée inutilement)
    mon_flux >> nonzero_elem; //lit le nombre d'élements non nuls

    // Définition de la la matrice A en sparse.
    A.resize(N,N);
    vector<Triplet<double>> liste_elem;
    for (int i = 0; i < nonzero_elem; i++)
    {
      mon_flux >> ligne;
      mon_flux >> colonne;
      mon_flux >> valeur;

      int li = atoi(ligne.c_str());
      int col = atoi(colonne.c_str());
      double val = atof(valeur.c_str());

      liste_elem.push_back({li-1,col-1,val});  //atoi pour passer de string à int et atof idem avec double
      if ((colonne != ligne) && sym) // dans le cas d'une matrice symétrique seulement la moitié des éléments sont dans le fichier texte
      {
        liste_elem.push_back({col-1,li-1,val});
      }
    }
    A.setFromTriplets(liste_elem.begin(),liste_elem.end());
    mon_flux.close();

    return A;
  }
/////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////
///////////////////// Gradient Conjugué ///////////////////////
///////////////////////////////////////////////////////////////
void GradientConj::Initialize(VectorXd x0, VectorXd b)
{
  // Initialisation
  _x = x0;
  _b = b;
  _r = _b - _A*_x;
  _p = _r;

  ofstream mon_flux; // Contruit un objet "ofstream"
  string name_file = ("/sol_"+to_string(_x.size())+"_Grad_Conj.txt");  //commande pour modifier le nom de chaque fichier
  mon_flux.open(name_file,ios::out);
}
///////////////////////////////////////////////////////////////
void GradientConj::Advance(VectorXd z)
{
  double alpha , gamma, stock_r;
  // Corps d'Advance
  stock_r = _r.dot(_r);    // On stock la norme de rk (nécessaire au calcul de gamma)
  alpha   =  _r.dot(_r)/(z.dot(_p));
  _x +=  alpha*_p ;
  _r += - alpha*z ;
  gamma = _r.dot(_r)/stock_r;
  _p = _r + gamma*_p;
}
///////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////
///////////////////// Residu Minimum ///////////////////////
////////////////////////////////////////////////////////////
void ResiduMin::Initialize(VectorXd x0, VectorXd b)
{
  // Initialisation
  _x = x0;
  _b = b;
  _r = _b - _A*_x;

  ofstream mon_flux; // Contruit un objet "ofstream"
  string name_file = ("/sol_"+to_string(_x.size())+"_Res_Min.txt");  //commande pour modifier le nom de chaque fichier
  mon_flux.open(name_file,ios::out);
}
///////////////////////////////////////////////////////////////
void ResiduMin::Advance(VectorXd z)
{
  double alpha(0.);
  // Corps d'Advance
  alpha = _r.dot(z)/z.dot(z);
  _x += alpha*_r;
  _r += - alpha*z;
}
///////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////
///////////////////// Gauss Seidel Symétrique ///////////////////////
/////////////////////////////////////////////////////////////////////
void SGS::Initialize(VectorXd x0, VectorXd b)
{
  // Initialisation
  _x = x0;
  _b = b;
  _r = _b - _A*_x;
  // Formation de L et U
  SparseMatrix<double> U, L, D;
  U.resize(_A.rows(), _A.cols()), L.resize(_A.rows(),_A.cols()), D.resize(_A.rows(),_A.cols());
  _U.resize(_A.rows(), _A.cols()), _L.resize(_A.rows(), _A.cols());
  U = _A;
  L = _A;
  for (int i=0; i<_A.rows(); i++)
  {
    for (int j=0; j<_A.cols(); j++)
    {
      if (i>j)
        {U.coeffRef(i,j) = 0.;}
      else if (j>i)
        {L.coeffRef(i,j) = 0.;}
      else
      {D.coeffRef(i,i) = 1./_A.coeffRef(i,i);}
    }
  }
  _L = L*D;
  _U = U;
  // On calcule y qui sera le même tout le long de la boucle "while"
  VectorXd y_bis(_x.size());       // On va Résoudre ici LUy = b
  y_bis = GetSolTriangInf(_L, b);  // Résolution L*y_bis = b
  _y = GetSolTriangSup(_U, y_bis); // Résolution U*y = b

  ofstream mon_flux; // Contruit un objet "ofstream"
  string name_file = ("/sol_"+to_string(_x.size())+"_SGS.txt");  //commande pour modifier le nom de chaque fichier
  mon_flux.open(name_file,ios::out);
}
/////////////////////////////////////////////////////////////////////
void SGS::Advance(VectorXd z)
{
  VectorXd  w(_x.size()), w_bis(_x.size());
  // Corps d'Advance
  w_bis = GetSolTriangInf(_L, z);  // Résolution de L*w_bis = z (avec z = A*xk)
  w = GetSolTriangSup(_U, w_bis);  // Résolution de U*w = w_bis
  _r = _b - z;    // On calcule d'abord le résidu avec xk et non xk+1 de manière à enlever un produit matrice/vecteur
  _x += - w + _y;  // et donc de gagner considérablement en temps de calcul quitte à faire un tour de boucle supplémentaire
}
/////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////
///////////////////// GMRes ///////////////////////
///////////////////////////////////////////////////
Gmres::Gmres(int Krylov)
{_Krylov = Krylov;}     // On affecte la valeur de la dimension de l'espace de Krylov

void Gmres::Initialize(VectorXd x0, VectorXd b)
{
  // Initialisation
  _x = x0;
  _b = b;
  _r = _b - _A*_x;
  _beta = _r.norm();

  ofstream mon_flux; // Contruit un objet "ofstream"
  string name_file = ("/sol_"+to_string(_x.size())+"_GMRes.txt");  //commande pour modifier le nom de chaque fichier
  mon_flux.open(name_file,ios::out);
}
///////////////////////////////////////////////////
void Gmres::Arnoldi( SparseMatrix<double> A , VectorXd v)
{
  //dimension de l'espace
  int m = _Krylov;
  // Initialisation
  _Vm.resize(v.size(), m+1);    // Matrice Vm de l'espace de Krylov
  vector< SparseVector<double> > Vm ;  // Vecteur utile au stock des valeurs
  SparseVector<double> v1 , s1 ;
  s1.resize(v.size());
  _Hm.resize( m+1 , m );   // Matrice de Hessenberg
  _Hm.setZero();
  vector< SparseVector<double> > z;
  z.resize(_Krylov);
  Vm.resize(m+1);

  v1 = v.sparseView();
  Vm[0]= v1/v1.norm(); // On affecte la premier vecteur de la base de Krylov à Vm+1
  // Corps de l'algo d'Arnoldi
  for (int j=0 ; j<m ; j++)
    {SparseVector<double> Av = A*Vm[j];
      s1.setZero();
        for(int i=0 ; i<j+1 ; i++)
        {
          _Hm.coeffRef(i,j) = Av.dot(Vm[i]);
          s1 = s1 + _Hm.coeffRef(i,j)*Vm[i];
        }
      z[j] = Av - s1;
      _Hm.coeffRef(j+1,j) = z[j].norm();
    // Si il y a un zéro sur la sous diagonale
    if(_Hm.coeffRef(j+1,j) == 0.)
    {   exit(0);}

    Vm[j+1] = z[j]/_Hm.coeffRef(j+1,j); // Afection du J ième vecteur de la base de Krylov
  }
  // Affectation des valeurs à la matrice Vm+1
  for(int i=0; i<v.size(); i++)
  {
    for (int j=0; j<m+1; j++)
    {_Vm.coeffRef(i,j) = Vm[j].coeffRef(i);}
  }
}
///////////////////////////////////////////////////
void Gmres::Givens(SparseMatrix<double> Hm)
{
  // Initialisation
  _Rm = Hm;
  _Qm.resize(Hm.rows(), Hm.rows());
  _Qm.setIdentity();
  double c(0.), s(0.), u(0.), v(0.);
  SparseMatrix<double> Rotation_transposee(Hm.rows(), Hm.rows());

  // Traitement de R
  for (int i=0; i<Hm.rows()-1; i++)
  {
    // Formation des coefficients de rotation
    Rotation_transposee.setIdentity();
    c = _Rm.coeffRef(i,i)/sqrt(_Rm.coeffRef(i,i)*_Rm.coeffRef(i,i) + _Rm.coeffRef(i+1,i)*_Rm.coeffRef(i+1,i));
    s = -_Rm.coeffRef(i+1,i)/sqrt(_Rm.coeffRef(i,i)*_Rm.coeffRef(i,i) + _Rm.coeffRef(i+1,i)*_Rm.coeffRef(i+1,i));
    Rotation_transposee.coeffRef(i,i) = c;
    Rotation_transposee.coeffRef(i+1,i+1) = c;
    Rotation_transposee.coeffRef(i+1,i) = -s;
    Rotation_transposee.coeffRef(i,i+1) = s;
    // Mise à jour de R
    for (int j=i; j<Hm.cols(); j++)
    {
      u = _Rm.coeffRef(i,j);
      v = _Rm.coeffRef(i+1,j);
      _Rm.coeffRef(i,j) = c*u - s*v;
      _Rm.coeffRef(i+1,j) = s*u + c*v;
    }
    // Mise à jour de Q
    _Qm = _Qm*Rotation_transposee;
  }
}
///////////////////////////////////////////////////
const SparseMatrix<double> & Gmres::GetVm() const
{
  return _Vm;  // On récupère Vm
}
///////////////////////////////////////////////////
const SparseMatrix<double> & Gmres::GetHm() const
{
  return _Hm;  // On récupère Hm
}
///////////////////////////////////////////////////
void Gmres::Advance(VectorXd z)
{
  // Initialisation
  VectorXd gm_barre(_Qm.rows()), gm(_Krylov), y(_Krylov), vect(_Qm.rows());
  SparseMatrix<double> Rm_pas_barre(_Rm.cols(), _Rm.cols());
  SparseMatrix<double> Vm;
  Vm.resize(z.size(), _Krylov);
  gm.setZero();
  gm_barre.setZero();
  vect.setZero();

  for (int i=0; i<_Qm.rows(); i++)
  {
    gm_barre[i] = _Qm.coeffRef(0,i); // On forme le vecteur gm_barre
    vect[i] = _Qm.coeffRef(i,_Krylov); // Vecteur utile pour la mise à jour de rk
  }

  gm_barre = z.norm()*gm_barre;  // gm_barre recoit sa valeur finale

  for (int i=0; i<_Krylov; i++)
  {
    gm[i] = gm_barre[i];   // On extrait gm à partir de gm_barre (on rejette le dernier terme de gm_barre)

    for (int j=0; j<_Krylov; j++)
    {
      Rm_pas_barre.coeffRef(i,j) = _Rm.coeffRef(i,j); // On extrait Rm (c'est Rm_barre sans la dernière ligne)
    }
    for (int j=0; j<Vm.rows(); j++)
    {
      Vm.coeffRef(j,i) = _Vm.coeffRef(j,i); // On extrait Vm à partir de Vm+1
    }
  }

  y = GetSolTriangSup(Rm_pas_barre, gm); // Résolution de Rm*y = gm

  _x = _x + Vm*y; // Mise à jour de xk

  _r = gm_barre[_Krylov]*_Vm*vect;  // Mise à jour de rk

  _beta = abs(gm_barre[_Krylov]);  // On extrait la norme de rk
}
///////////////////////////////////////////////////
const double & Gmres::GetNorm() const
{
  return _beta;  // Récupère la norme du résidu (rk)
}
///////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////
///////// Gradient Conjugué Préconditionné ////////////////////////
///////////////////////////////////////////////////////////////////

void GradientConPrecond::Initialize(Eigen::VectorXd x0, Eigen::VectorXd b)
{
  _x = x0;

  ofstream mon_flux; // Contruit un objet "ofstream"
  string name_file = ("/sol_"+to_string(_x.size())+"_grad_conj_precond.txt");  //commande pour modifier le nom de chaque fichier
  mon_flux.open(name_file,ios::out);
//construit le preconditionneur SGS
  _D.resize(_x.size(),_x.size()), _D_inv.resize(_x.size(),_x.size()), _E.resize(_x.size(),_x.size()), _F.resize(_x.size(),_x.size());
  _M_grad.resize(_x.size(),_x.size());
  _D.setZero(); _F.setZero(); _E.setZero();
  for (int i =0; i<_x.size(); i++)
  {
    _D.coeffRef(i,i) = _A.coeffRef(i,i);
    _D_inv.coeffRef(i,i) = 1./_A.coeffRef(i,i);

    for(int j = 0; j<_x.size(); j++)
    {
      if (j>i)
      {_F.coeffRef(i,j) = - _A.coeffRef(i,j);}

      else if (j<i)
      {_E.coeffRef(i,j) = - _A.coeffRef(i,j);}
    }

  }
  _M_grad = (_D - _E)*_D_inv*(_D - _F);
  _b = b;
  _r = _b - _A*_x;
  _p = _r;                   // utile pour le GradientConj


}

const SparseMatrix<double> & GradientConPrecond::Get_M() const
{

  return _M_grad;
}


void GradientConPrecond::Advance(Eigen::VectorXd z)
{
  double alpha , gamma, stock_r;

  stock_r = _r.dot(_r);

  alpha   =  stock_r/(z.dot(_p));
  _x +=  alpha*_p ;
  _r += - alpha*z ;
  gamma = _r.dot(_r)/stock_r;
  _p = _r + gamma*_p;
}
///////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////
///////////////////// GMRES Preconditionne ///////////////////////
///////////////////////////////////////////////////////////////////


///////////////////// GMRes Preconditionne ///////////////////////
Gmresprecond::Gmresprecond(int Krylov)
{_Krylov = Krylov;}
/////////////////////////////////////////////
void Gmresprecond::Initialize(VectorXd x0, VectorXd b)
{
  _x = x0;
  _b = b;
  _r = _b - _A*_x;
  _beta = _r.norm();

  _D.resize(_x.size(),_x.size()), _D_inv.resize(_x.size(),_x.size()), _E.resize(_x.size(),_x.size()), _F.resize(_x.size(),_x.size());
  _M_sgs.resize(_x.size(),_x.size());
  _D.setZero(); _F.setZero(); _E.setZero();
  for (int i =0; i<_x.size(); i++)
      {
          _D.coeffRef(i,i) = _A.coeffRef(i,i);
          _D_inv.coeffRef(i,i) = 1./_A.coeffRef(i,i);

   for(int j = 0; j<_x.size(); j++)
   {
     if (j>i)
     {_F.coeffRef(i,j) = - _A.coeffRef(i,j);}

     else if (j<i)
     {_E.coeffRef(i,j) = - _A.coeffRef(i,j);}
   }

      }
  _M_sgs = (_D - _E)*_D_inv*(_D - _F);   // Matrice de préconditionnement
  // On en fait la décomposition LU
  _solver1.compute(_M_sgs);

  ofstream mon_flux; // Contruit un objet "ofstream"
  string name_file = ("/sol_"+to_string(_x.size())+"_Gmres.txt");  //commande pour modifier le nom de chaque fichier
  mon_flux.open(name_file,ios::out);
}
////////////////////////////////////////////////////
void Gmresprecond::Arnoldi( SparseMatrix<double> A , VectorXd v)
{
  //dimension de l'espace
  int m = _Krylov;
  // Initialisation
  _Vm.resize(v.size(), m+1);    // Matrice Vm de l'espace de Krylov
  _Wm.resize(v.size(), m+1);    // Matrice Wm préconditionnée de l'espace de Krylov
  vector< SparseVector<double> > Vm ;  // Vecteur utile au stock des valeurs
  SparseVector<double> v1 , s1 ;
  s1.resize(v.size());
  _Hm.resize( m+1 , m );   // Matrice de Hessenberg
  _Hm.setZero();
  vector< SparseVector<double> > z;
  vector< SparseVector<double> > Wm;  // Matrice contenant la résolution de Wj = M⁽-1)*Vj
  z.resize(_Krylov);
  Vm.resize(m+1);
  Wm.resize(m+1);

  v1 = v.sparseView();
  Vm[0]= v1/v1.norm(); // On affecte la premier vecteur de la base de Krylov à Vm+1

  // Corps de l'algo d'Arnoldi
  for (int j=0 ; j<m ; j++)
  {
    Wm[j] = _solver1.solve(Vm[j]).sparseView();
    SparseVector<double> AMv = A*Wm[j];
    s1.setZero();
    for(int i=0 ; i<j+1 ; i++)
    {
      _Hm.coeffRef(i,j) = AMv.dot(Vm[i]);
      s1 = s1 + _Hm.coeffRef(i,j)*Vm[i];
    }
    z[j] = AMv - s1;
    _Hm.coeffRef(j+1,j) = z[j].norm();
    // Si il y a un zéro sur la sous diagonale
    if(_Hm.coeffRef(j+1,j) == 0.)
    {   exit(0);}

    Vm[j+1] = z[j]/_Hm.coeffRef(j+1,j); // Afection du J ième vecteur de la base de Krylov
  }
  Wm[m] = _solver1.solve(Vm[m]).sparseView();

  // Affectation des valeurs à la matrice Vm+1
  for(int i=0; i<v.size(); i++)
  {
    for (int j=0; j<m+1; j++)
    {
      _Vm.coeffRef(i,j) = Vm[j].coeffRef(i);
      _Wm.coeffRef(i,j) = Wm[j].coeffRef(i);
    }
  }
}
///////////////////////////////////////////////////
void Gmresprecond::Givens(SparseMatrix<double> Hm)
{
  // Initialisation
  _Rm = Hm;
  _Qm.resize(Hm.rows(), Hm.rows());
  _Qm.setIdentity();
  double c(0.), s(0.), u(0.), v(0.);
  SparseMatrix<double> Rotation_transposee(Hm.rows(), Hm.rows());

  // Traitement de R
  for (int i=0; i<Hm.rows()-1; i++)
  {
    // Formation des coefficients de rotation
    Rotation_transposee.setIdentity();
    c = _Rm.coeffRef(i,i)/sqrt(_Rm.coeffRef(i,i)*_Rm.coeffRef(i,i) + _Rm.coeffRef(i+1,i)*_Rm.coeffRef(i+1,i));
    s = -_Rm.coeffRef(i+1,i)/sqrt(_Rm.coeffRef(i,i)*_Rm.coeffRef(i,i) + _Rm.coeffRef(i+1,i)*_Rm.coeffRef(i+1,i));
    Rotation_transposee.coeffRef(i,i) = c;
    Rotation_transposee.coeffRef(i+1,i+1) = c;
    Rotation_transposee.coeffRef(i+1,i) = -s;
    Rotation_transposee.coeffRef(i,i+1) = s;
    // Mise à jour de R
    for (int j=i; j<Hm.cols(); j++)
    {
      u = _Rm.coeffRef(i,j);
      v = _Rm.coeffRef(i+1,j);
      _Rm.coeffRef(i,j) = c*u - s*v;
      _Rm.coeffRef(i+1,j) = s*u + c*v;
    }
    // Mise à jour de Q
    _Qm = _Qm*Rotation_transposee;
  }
}
///////////////////////////////////////////////////
void Gmresprecond::Advance(VectorXd z)
{
  // Initialisation
  VectorXd gm_barre(_Qm.rows()), gm(_Krylov), y(_Krylov), vect(_Qm.rows());
  SparseMatrix<double> Rm_pas_barre(_Rm.cols(), _Rm.cols());
  SparseMatrix<double> Wm;
  Wm.resize(z.size(), _Krylov);  // Wm = M⁻¹*Vm
  gm.setZero();
  gm_barre.setZero();
  vect.setZero();

  for (int i=0; i<_Qm.rows(); i++)
  {
    gm_barre[i] = _Qm.coeffRef(0,i); // On forme le vecteur gm_barre
    vect[i] = _Qm.coeffRef(i,_Krylov); // Vecteur utile pour la mise à jour de rk
  }

  gm_barre = z.norm()*gm_barre;  // gm_barre recoit sa valeur finale

  for (int i=0; i<_Krylov; i++)
  {
    gm[i] = gm_barre[i];   // On extrait gm à partir de gm_barre (on rejette le dernier terme de gm_barre)

    for (int j=0; j<_Krylov; j++)
    {
      Rm_pas_barre.coeffRef(i,j) = _Rm.coeffRef(i,j); // On extrait Rm (c'est Rm_barre sans la dernière ligne)
    }
    for (int j=0; j<Wm.rows(); j++)
    {
      Wm.coeffRef(j,i) = _Wm.coeffRef(j,i); // On extrait Vm à partir de Vm+1
    }
  }

  y = GetSolTriangSup(Rm_pas_barre, gm); // Résolution de Rm*y = gm

  _x = _x + Wm*y; // Mise à jour de xk

  _r = gm_barre[_Krylov]*_Vm*vect;  // Mise à jour de rk

  _beta = abs(gm_barre[_Krylov]);  // On extrait la norme de rk
}
///////////////////////////////////////////////////
const double & Gmresprecond::GetNorm() const
{
  return _beta;  // Récupère la norme du résidu (rk)
}
///////////////////////////////////////////////////
const SparseMatrix<double> & Gmresprecond::GetHm() const
{
  return _Hm;  // Récupère la norme du résidu (rk)
}
///////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////
///////////////////// Fonctions hors classe ///////////////////////
///////////////////////////////////////////////////////////////////
VectorXd GetSolTriangSup(SparseMatrix<double> U, VectorXd b)
{
  VectorXd solution(U.rows());

  for (int i=0; i<U.rows(); i++)
  {
    solution[U.rows()-i-1] = b[U.rows()-i-1];
    for (int j=U.rows()-i; j<U.rows(); j++)
    {
      solution[U.rows()-1-i] = solution[U.rows()-1-i] - U.coeffRef(U.rows()-1-i,j)*solution[j];
    }
    solution[U.rows()-1-i] = solution[U.rows()-1-i]/U.coeffRef(U.rows()-1-i,U.rows()-1-i);
  }
  return solution;
}
///////////////////////////////////////////////////////////////////
VectorXd GetSolTriangInf(SparseMatrix<double> L, VectorXd b)
{
  VectorXd solution(L.rows());

  for (int i=0; i<L.rows(); i++)
  {
    solution[i] = b[i];
    for (int j=0; j<i; j++)
    {
      solution[i] = solution[i] - L.coeffRef(i,j)*solution(j);
    }
    solution[i] = solution[i]/L.coeffRef(i,i);
  }
  return solution;
}
/////////////////////////////////////////////////////////////////////
double Verif_norme( Eigen::VectorXd x, Eigen::VectorXd b, Eigen::SparseMatrix<double> A)
  {return (b-A*x).norm();}


#define _Meth_Iterative_H
#endif
