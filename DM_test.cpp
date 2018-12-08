#ifndef _Meth_Iterative_H

#include "DM_test.h"  // Inclure le fichier .h

using namespace std;
using namespace Eigen;

// constructeur par défaut
MethIterative::MethIterative()
{}

// Destructeur
MethIterative::~MethIterative()
{}

void MethIterative::MatrixInitialize(SparseMatrix<double> A)
{  _A.resize(A.rows(), A.cols());
    _A = A;}

//////////////////// RESIDU MINIMUM///////////////////////////////////////////////////////////////////////////////////
// initialisation des données
void ResiduMin::Initialize(VectorXd x0, VectorXd b)
{_x = x0;
  _b = b;
  _r = _b - _A*_x;

  ofstream mon_flux; // Contruit un objet "ofstream"
  string name_file = ("/sol_"+to_string(_x.size())+"_res_min.txt");  //commande pour modifier le nom de chaque fichier
  mon_flux.open(name_file,ios::out);}

void ResiduMin::Advance(VectorXd z)
{
  double alpha(0.);
  alpha = _r.dot(z)/z.dot(z);

  _x += alpha*_r;
  _r += - alpha*z;
}

const VectorXd & MethIterative::GetIterateSolution() const
{return _x;}

const VectorXd & MethIterative::GetResidu() const
{return _r;}

//////GRADIENT CONJUGUE ////////////////////////////////////////////////////////////

void GradientConj::Initialize(VectorXd x0, VectorXd b)
{_x = x0;
  _b = b;
  _r = _b - _A*_x;
  _p = _r;                   // utile pour le GradientConj
  ofstream mon_flux; // Contruit un objet "ofstream"
  string name_file = ("/sol_"+to_string(_x.size())+"_grad_conj.txt");  //commande pour modifier le nom de chaque fichier
  mon_flux.open(name_file,ios::out);}

void GradientConj::Advance(VectorXd z)
{double alpha , gamma, stock_r;

  stock_r = _r.dot(_r);
  alpha   =  _r.dot(_r)/(z.dot(_p));
  _x +=  alpha*_p ;
  _r += - alpha*z ;
  gamma = _r.dot(_r)/stock_r;
  _p = _r + gamma*_p;}

const VectorXd & MethIterative::Getp() const
{  return _p;}



/////SGS/////////////////////////////////////////////////////////////////////////////

void SGS::Initialize(VectorXd x0, VectorXd b)
{_x = x0;
  _b = b;
  _r = _b - _A*_x;
  SparseMatrix<double> U, L, D;
  U.resize(_A.rows(), _A.cols()), L.resize(_A.rows(),_A.cols()), D.resize(_A.rows(),_A.cols());
  _U.resize(_A.rows(), _A.cols()), _L.resize(_A.rows(), _A.cols());

  U = _A;
  L = _A;
  _U = _A;
  _L = _A;
  for (int i=0; i<_A.rows(); i++)
  {
    for (int j=0; j<_A.cols(); j++)
    {
      if (i>j)
        {U.coeffRef(i,j) = 0.;
        _U.coeffRef(i,j) = 0.;}
      else if (j>i)
        {L.coeffRef(i,j) = 0.;
        _L.coeffRef(i,j) = 0.;}
      else
        {D.coeffRef(i,i) = 1./_A.coeffRef(i,i);
        _L.coeffRef(i,i) = 1./_A.coeffRef(i,i);}
    }
  }
  _M = L*D*U;
  _L = L*D;
  _U = U;
  // cout << _M << endl;
  // cout << "--------" << endl;
  // cout << _L*_U << endl;
  /*
  SparseLU< SparseMatrix<double>> lu1;

  lu1.analyzePattern(_M) ;
  lu1.factorize(_M);
  _y = lu1.solve(_b);

  SparseLU< SparseMatrix<double> > lu2;
  lu2.analyzePattern(_L*_U);
  lu2.factorize(_L*_U);
  _y = lu2.solve(_b);
  */
  VectorXd y_bis(_x.size());
  y_bis = GetSolTriangInf(_L, b);
  _y = GetSolTriangSup(_U, y_bis);}


void SGS::Advance(VectorXd z)
{
  VectorXd  w(_x.size()), w_bis(_x.size());

  // SparseLU< SparseMatrix<double> > lu1;
  // lu1.analyzePattern(_M) ;
  // lu1.factorize(_M);
  // y = lu1.solve(_b);
  /*
  SparseLU< SparseMatrix<double> > lu2;
  lu2.analyzePattern(_M) ;
  lu2.factorize(_M);
  w = lu2.solve(z);
  */
  w_bis = GetSolTriangInf(_L, z);
  w = GetSolTriangSup(_U, w_bis);
  _r = _b - z;
  _x += - w + _y;

}



/////////////////////// GMRES ////////////////////////////////////////

void Gmres::Initialize(VectorXd x0, VectorXd b)
{
  _x = x0;
  _b = b;
  _r = _b - _A*_x;
  _beta = _r.norm();

  ofstream mon_flux; // Contruit un objet "ofstream"
  string name_file = ("/sol_"+to_string(_x.size())+"_grad_conj.txt");  //commande pour modifier le nom de chaque fichier
  mon_flux.open(name_file,ios::out);
}


void Gmres::Arnoldi( SparseMatrix<double> A , VectorXd v)
{
  //dimension de l'espace
  int m = v.size();

  _Vm.resize(m, m+1);

  vector< SparseVector<double> > Vm ;
  SparseVector<double> v1 , s1 ;
  s1.resize(v.size());
  v1 = v.sparseView();

  _Hm.resize( m+1 , m );
  _Hm.setZero();

  vector< SparseVector<double> > z;
  z.resize(v.size());
  Vm.resize(m+1);
  Vm[0]= v1/v1.norm();
  cout<<"premiere boucle for avant"<<endl;
  for (int j=0 ; j<m ; j++)
    {SparseVector<double> Av = A*Vm[j];
      s1.setZero();
    //cout<<"2eme boucle for avant"<<endl;
        for(int i=0 ; i<j+1 ; i++)
        {

          _Hm.coeffRef(i,j) = Av.dot(Vm[i]);
          s1 +=  _Hm.coeffRef(i,j)*Vm[i];

        }
      z[j] = Av - s1;
      _Hm.coeffRef(j+1,j) = z[j].norm();
    if(_Hm.coeffRef(j+1,j) == 0.)
    {   break;}
    Vm[j+1] = z[j]/_Hm.coeffRef(j+1,j);
  }
  cout<<"3eme boucle for avant"<<endl;

  for(int i=0; i<m; i++)
  {
    for (int j=0; j<m+1; j++)
    {_Vm.coeffRef(i,j) = Vm[j].coeffRef(i);}
  }

}


void Gmres::Givens(SparseMatrix<double> Hm)
{

// J'ai mis R et Q de taille carré...
  _Rm = Hm;
  //cout << Hm << endl;
  _Qm.resize(Hm.rows(), Hm.rows());
  //cout << "QM " << _Qm.rows() << _Qm.cols() << endl;
  _Qm.setIdentity();
  double c(0.), s(0.), u(0.), v(0.);
  SparseMatrix<double> Rotation_transposee(Hm.rows(), Hm.rows());

  for (int i=0; i<Hm.rows()-1; i++)
  {
    Rotation_transposee.setIdentity();
    c = _Rm.coeffRef(i,i)/sqrt(_Rm.coeffRef(i,i)*_Rm.coeffRef(i,i) + _Rm.coeffRef(i+1,i)*_Rm.coeffRef(i+1,i));
    s = -_Rm.coeffRef(i+1,i)/sqrt(_Rm.coeffRef(i,i)*_Rm.coeffRef(i,i) + _Rm.coeffRef(i+1,i)*_Rm.coeffRef(i+1,i));
    Rotation_transposee.coeffRef(i,i) = c;
    Rotation_transposee.coeffRef(i+1,i+1) = c;
    Rotation_transposee.coeffRef(i+1,i) = -s;
    Rotation_transposee.coeffRef(i,i+1) = s;

    for (int j=i; j<Hm.cols(); j++)
    {
      u = _Rm.coeffRef(i,j);
      v = _Rm.coeffRef(i+1,j);
      _Rm.coeffRef(i,j) = c*u - s*v;
      _Rm.coeffRef(i+1,j) = s*u + c*v;
      //if (j == i)
      //{cout << "ici " << _Rm.coeffRef(i+1,j) << endl;}
    }

    _Qm = _Qm*Rotation_transposee;
    //cout << "Rm rows / cols" << _Rm.rows() << " " << _Rm.cols() << endl;
  }
    // cout << "_vm = "<< _Vm << endl;
}


const SparseMatrix<double> & Gmres::GetVm() const
{
  return _Vm;
}

const SparseMatrix<double> & Gmres::GetHm() const
{
  return _Hm;
}

void Gmres::Advance(VectorXd z)
{
  VectorXd gm_barre(_Qm.rows()), gm(z.size()), y(z.size()), vect(_Qm.rows());
  SparseMatrix<double> Rm_pas_barre(z.size(), z.size());
  SparseMatrix<double> Vm;
  Vm.resize(z.size(), z.size());

  gm.setZero();
  gm_barre.setZero();
  vect.setZero();

  for (int i=0; i<_Qm.rows(); i++)
  {
    gm_barre[i] = _Qm.coeffRef(0,i);
    vect[i] = _Qm.coeffRef(i,z.size());
  }

  gm_barre = z.norm()*gm_barre;

  for (int i=0; i<z.size(); i++)
  {
    gm[i] = gm_barre[i];
    for (int j=0; j<z.size(); j++)
    {
      Rm_pas_barre.coeffRef(i,j) = _Rm.coeffRef(i,j);
      Vm.coeffRef(i,j) = _Vm.coeffRef(i,j);
    }
  }

  y = GetSolTriangSup(Rm_pas_barre, gm);

  _x = _x + Vm*y;

  _r = gm_barre[z.size()]*_Vm*vect;
  _beta = abs(gm_barre[z.size()]);
    //cout << " r = " << _r << endl;
    cout << "gm+1 " << gm_barre[_r.size()] << endl;
    cout <<"norme de r " << _r.norm() << endl;
  // cout << "juste après l'affectation de r dans advance" << endl;
  //
  // cout << "_vm = "<< _Vm << endl;
}
const double & Gmres::GetNorm() const
{
  return _beta;
}


///////////////////// Fonctions hors classe ///////////////////////
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



SparseMatrix<double>  MethIterative::create_mat(const string name_file_read, bool sym)
  {
    int N(0.);
    Eigen::SparseMatrix<double> A ;
    ifstream mon_flux(name_file_read);

    string ligne, colonne, valeur;
    getline(mon_flux,ligne); //lit la première ligne qui ne nous intéresse pas

    mon_flux >> N; //lit le premier mot de la ligne 2 correspond au nombre de lignes

    int nonzero_elem;
    mon_flux >> colonne; //lit le nombre de colonnes (valeur stockée inutilement)
    mon_flux >> nonzero_elem; //lit le nombre d'élements non nuls

    // Définition de la la matrice A.
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



// Écrit un fichier avec la solution
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

void  MethIterative::Get_norme_sol()
{ VectorXd Xverif(_x.size());
  Xverif = _b-_A*_x;

cout<<"norme de (b-Ax)  = "<<Xverif.norm()<<endl;}



#define _Meth_Iterative_H
#endif
