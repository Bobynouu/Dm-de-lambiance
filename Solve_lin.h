#ifndef _Meth_Iterative_H

#include <vector>
#include <iostream>
#include <string>
#include <cmath>
#include <fstream>
#include <sstream>
#include <complex>
#include "Dense"
#include "Sparse"
#include "Eigen"


class MethIterative
{
  protected:
    Eigen::VectorXd _x;  // solution itérée
    Eigen::VectorXd _r;  // vecteur résidu
    Eigen::VectorXd _b;  // vecteur du second membre
    Eigen::SparseMatrix<double> _A; // matrice du systeme
    Eigen::VectorXd _p;   // nécessaire à l'algorithme du Gradient Conjugue

  public:
    // constructeur à partir d'une matrice A dense donnée
    MethIterative();
    // destructeur
    virtual ~MethIterative();
    // crée une matrice à partir d'un fichier de MatrixMarket
    Eigen::SparseMatrix<double> create_mat(const std::string name_file_read, bool sym);
    // initialise une Matrice
    void MatrixInitialize(Eigen::SparseMatrix<double> A);
    // initialise les données
    virtual void Initialize(Eigen::VectorXd x0, Eigen::VectorXd b)=0;
    // exécute une itération
    virtual void Advance(Eigen::VectorXd z) = 0;
    // récupère le vecteur xk
    const Eigen::VectorXd & GetIterateSolution() const;
    // Récupère le Résidu
    const Eigen::VectorXd & GetResidu() const;
    // Récupère le vecteur p
    virtual const Eigen::VectorXd & Getp() const;
    // Sauvgarde la norme du résidu dans un fichier
    void saveSolution(int N , std::string name_file ,  int n_iter , double residu);


};

class ResiduMin : public MethIterative
{
  public:
    // exécute une itération
    void Advance(Eigen::VectorXd z);
    // initialise les données
    void Initialize(Eigen::VectorXd x0, Eigen::VectorXd b);
};

class GradientConj : public MethIterative
{
  public:
    // exécute une itération
    void Advance(Eigen::VectorXd z);
    // initialise les données
    void Initialize(Eigen::VectorXd x0, Eigen::VectorXd b);
};

class SGS : public MethIterative
{
  private:
    Eigen::SparseMatrix<double> _L;   // Matrice égale à la partie triangulaire inférieure de M
    Eigen::SparseMatrix<double> _U;   // Matrice égale à la partie triangulaire supérieure de M
    Eigen::VectorXd _y;               // Stock un vecteur dans SGS
  public:
    // initialise les données
    void Initialize(Eigen::VectorXd x0, Eigen::VectorXd b);
    // exécute une itération
    void Advance(Eigen::VectorXd z);
};

class Gmres : public MethIterative
{
  private:
    Eigen::SparseMatrix<double> _Vm ;  // Matrice de l'espace de Krylov via Arnoldi
    Eigen::SparseMatrix<double> _Hm;   // Matrice d'Hessenberg via Arnoldi
    Eigen::SparseMatrix<double> _Qm;   // Matrice Q de la décomposition QR via Givens
    Eigen::SparseMatrix<double> _Rm;   // Matrice R de la décomposition QR via Givens
    double _beta;   // Stock la norme du résidu après une itération
    int _Krylov;    // Dimension de l'espace de Krylov
  public:
    // Récupère la matrice d'Hessenberg
    const Eigen::SparseMatrix<double> & GetHm() const;
    // Récupère la matrice de l'espace de Krylov
    const Eigen::SparseMatrix<double> & GetVm() const;
    // Récupère la norme du résidu
    const double & GetNorm() const;
    // exécute une itération
    void Advance(Eigen::VectorXd z);
    // initialise les données
    void Initialize(Eigen::VectorXd x0, Eigen::VectorXd b);
    // Exécute l'algorithme d'Arnoldi à partir de A et le résidu, on obtient une matrice d'Hessenberg
    void Arnoldi(Eigen::SparseMatrix<double> A, Eigen::VectorXd v);
    // Exécute la décomposition QR d'une matrice d'Hessenberg via Givens
    void Givens(Eigen::SparseMatrix<double> Hm);

};

// Résolution d'un système linéaire pour une matrice triangulaire supérieure
Eigen::VectorXd GetSolTriangSup(Eigen::SparseMatrix<double> U, Eigen::VectorXd b);
// Résolution d'un système linéaire pour une matrice triangulaire inférieure
Eigen::VectorXd GetSolTriangInf(Eigen::SparseMatrix<double> L, Eigen::VectorXd b);
//Calcul de la norme de b-Ax pour vérifier la solution obtenue
double Verif_norme( Eigen::VectorXd x, Eigen::VectorXd b, Eigen::SparseMatrix<double> A);

#define _Meth_Iterative_H
#endif
