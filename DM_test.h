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
    Eigen::VectorXd _p;
    Eigen::SparseMatrix<double> _A; // matrice du systeme

  public:
    // constructeur à partir d'une matrice A dense donnée
    MethIterative();
    // destructeur
    ~MethIterative();
    // initialise une Matrice
    void MatrixInitialize(Eigen::SparseMatrix<double> A);
    // initialise les données
    virtual void Initialize(Eigen::VectorXd x0, Eigen::VectorXd b)=0;
    // initialise la matrice A venant d'un fichier
    //SparseMatrix<double> MatrixInitialize(const std::string name_file_read, bool sym);
    // exécute une itération
    virtual void Advance(Eigen::VectorXd z) = 0;
    // récupère le vecteur xk et le vecteur rk
    const Eigen::VectorXd & GetIterateSolution() const;
    const Eigen::VectorXd & GetResidu() const;
    const Eigen::VectorXd & Getp() const;
    void saveSolution(int N , std::string name_file ,  int n_iter , double residu);
    Eigen::SparseMatrix<double> create_mat(const std::string name_file_read, bool sym);

};

class ResiduMin : public MethIterative
{
  public:
    void Advance(Eigen::VectorXd z);
    void Initialize(Eigen::VectorXd x0, Eigen::VectorXd b);
};

class GradientConj : public MethIterative
{
  public:
    void Advance(Eigen::VectorXd z);
    void Initialize(Eigen::VectorXd x0, Eigen::VectorXd b);
};

class SGS : public MethIterative
{
  private:
    Eigen::SparseMatrix<double> _M;
    Eigen::VectorXd _y;
  public:
    void Initialize(Eigen::VectorXd x0, Eigen::VectorXd b);
    void Advance(Eigen::VectorXd z);
};

class Gmres : public MethIterative
{
  private:
    std::vector< Eigen::SparseVector<double> > _Vm ;
    Eigen::SparseMatrix<double> _Hm;
  public:
    void Advance(Eigen::VectorXd z);
    void Initialize(Eigen::VectorXd x0, Eigen::VectorXd b);
    void Arnoldi(Eigen::VectorXd v);
};



#define _Meth_Iterative_H
#endif
