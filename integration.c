#include <stdio.h>
#include <math.h>
#include <stdbool.h>

double integrand(int n, double args[n]);
double q_integrand_logisticQ(int n, double args[n]);
double q_integrand_logisticQ_malmquist(int n, double args[n]);
double q_integrand_logisticQ_malmquist_cutoff(int n, double args[n]);
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

double q_integrand_logisticQ_malmquist_cutoff(int n, double args[n])
{
    //unpack arguments
    double q = args[0];
    double gamma = args[1];
    double f_bin = args[2];
    double Pobs = args[3];
    double alpha = args[4];
    double beta = args[5];
    double lowq = args[6];
    double highq = args[7];
    double constant = 1;

    //make malmquist correction array
    int arr_size = args[8];
    double malm_pars[arr_size];
    double denominator = 0.0;
    double integral = 0.0;
    for (int i=0; i<arr_size; ++i)
    {
        malm_pars[i] = args[i+9];
        integral += malm_pars[i] * (1-gamma)/(1+i-gamma) * (pow(highq, 1+i-gamma) - pow(lowq, 1+i-gamma));
    }
    if (arr_size == 1) { Pobs = integral;}
    denominator = f_bin*integral + (1-f_bin)*Pobs;
    
    constant = (1-gamma) / (pow(highq, 1-gamma) - pow(lowq, 1-gamma));
    return f_bin*constant*pow(q, -gamma) / (1.0 + exp(-alpha*(q-beta))) * polynomial(q, arr_size, malm_pars)/denominator;
}


double q_integrand_logisticQ_malmquist_cutoff_old(int n, double args[n])
{
    //unpack arguments
    double q = args[0];
    double gamma = args[1];
    double alpha = args[2];
    double beta = args[3];
    double lowq = args[4];
    double highq = args[5];
    double constant = 1;

    //make malmquist correction array
    int arr_size = args[6];
    double malm_pars[arr_size];
    double denominator = 0.0;
    for (int i=0; i<arr_size; ++i)
    {
        malm_pars[i] = args[i+7];
        denominator += malm_pars[i] * (1-gamma)/(1+i-gamma);
    }

    constant = (1-gamma) / (denominator * (pow(highq, 1-gamma) - pow(lowq, 1-gamma)));
    return constant*pow(q, -gamma) / (1.0 + exp(-alpha*(q-beta))) * polynomial(q, arr_size, malm_pars);
}

double hist_integrand(int n, double args[n])
{
    //unpack arguments
    double q = args[0];
    double alpha = args[1];
    double beta = args[2];
    double f_bin = args[3];
    double Pobs = args[4];
    bool malmcorr = args[5];
    int n_bins = args[6];
    int n_malm = args[7];
    int i, j;
    double thetas[n_bins], bin_edges[n_bins+1], malm_pars[n_malm], malm_integrals[n_bins];
    double denom_integral = 0.0;
    double denom, gamma, Q;

    // Make theta array and bin edges
    for (i=0; i<n_bins; ++i) 
    {
        thetas[i] = args[8+i];
        bin_edges[i] = args[8+i+n_bins];
    }
    bin_edges[n_bins] = args[8+2*n_bins];

    // make malmquist correction array
    for (i=0; i<n_malm; ++i)
    {
        malm_pars[i] = args[8+i+2*n_bins+1];
    }
    for (i=0; i<n_bins; ++i);
    {
        malm_integrals[i] = 0.0;
        for (j=0; j<n_malm; ++j);
            malm_integrals[i] += malm_pars[j]/(j+1.0)*(pow(bin_edges[i+1], j+1) - pow(bin_edges[i], j+1));

        // Which bin is the requested q in?
        if (q > bin_edges[i] & q <= bin_edges[i+1]) gamma = thetas[i];
    }
    
    // Calculate denominator integral
    denom_integral = 0.0;
    for (i=0; i<n_bins; ++i)
    {
        denom_integral += thetas[i] * malm_integrals[i];
    }
    if (!malmcorr) Pobs = denom_integral;
    denom = f_bin*denom_integral + (1-f_bin)*Pobs;

    // Finally, calculate gamma
    gamma *= polynomial(q, n_malm, malm_pars) * f_bin / denom;

    Q = 1.0/(1.0 + exp(-alpha*(q-beta)));
    return Q*gamma;




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
