#include <stdio.h>
#include <math.h>

double integrand(int n, double args[n]);
double q_integrand_logisticQ(int n, double args[n]);
double q_integrand_logisticQ_malmquist(int n, double args[n]);
double polynomial(double x, int n, double pars[n]);
double integrand2(int n, double* args);
double Q(double q, double a, double e);

static const double SQ2PI = sqrt(2*M_PI);

double integrand(int n, double args[n])
{
    //unpack arguments
    double lna = args[0];
    double e = args[1];
    double q = args[2];
    double gamma = args[3];
    double mu = args[4];
    double sigma = args[5];
    double eta = args[6];
    
    // Calculate the integrand
    double Gamma_q = (1-gamma)*pow(q, -gamma);
    double Gamma_e = (1-eta)*pow(e, -eta);
    double Gamma_a = 1.0/(sigma*SQ2PI) * exp(-0.5*(lna-mu)*(lna-mu)/(sigma*sigma));
    double Q_val = Q(q, exp(lna), e);

    return Gamma_q*Gamma_e*Gamma_a*Q_val;    
}

double q_integrand_logisticQ(int n, double args[n])
{
    //unpack arguments
    double q = args[0];
    double gamma = args[1];
    double alpha = args[2];
    double beta = args[3];
    
    return (1-gamma)*pow(q, -gamma) / (1.0 + exp(-alpha*(q-beta)));
}

double q_integrand_logisticQ_malmquist(int n, double args[n])
{
    //unpack arguments
    double q = args[0];
    double gamma = args[1];
    double alpha = args[2];
    double beta = args[3];

    //make malmquist correction array
    int arr_size = args[4];
    double malm_pars[arr_size];
    double denominator = 0.0;
    for (int i=0; i<arr_size; ++i)
    {
        malm_pars[i] = args[i+5];
        denominator += malm_pars[i] * (1-gamma)/(1+i-gamma);
    }
    
    return (1-gamma)*pow(q, -gamma) / (1.0 + exp(-alpha*(q-beta))) * polynomial(q, arr_size, malm_pars)/denominator;
}

double polynomial(double x, int n, double pars[n])
{
    double retval = 0.0;
    for (int i=0; i<n; ++i)
    {
        retval += pars[i]*pow(x, i);
    }
    return retval;
}

double integrand2(int n, double* args)
{
    double lna = args[0];
    double e = args[1];
    double q = args[2];
    double gamma = args[3];
    double mu = args[4];
    double sigma = args[5];
    double eta = args[6];
    
    // Calculate the integrand
    double Gamma_q = (1-gamma)*pow(q, -gamma);
    double Gamma_e = (1-eta)*pow(e, -eta);
    double Gamma_a = 1.0/(sigma*SQ2PI) * exp(-0.5*(lna-mu)*(lna-mu)/(sigma*sigma));
    double Q_val = Q(q, exp(lna), e);

    return Gamma_q*Gamma_e*Gamma_a*Q_val;

}

double Q(double q, double a, double e)
{
    // Calculate the completeness fraction
    if (q > 0.1)
    {
      return 1.0;
    }
    else
    {
      return 0.0;
    }
}

//int main()
//{
//    double vals[7] = {1.0, 0.3, 0.4, 0.4, 5.7, 2.3, 0.7};
//    double *pvals = &vals;
//    double I = integrand2(7, pvals);
//    printf("Integral value = %8.7f\n", I);
//}
