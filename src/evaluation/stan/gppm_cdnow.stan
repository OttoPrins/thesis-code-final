data {
  int<lower=1> N;
  int<lower=1> T_calib;
  int<lower=T_calib> T_total;
  array[N, T_calib] int<lower=0> x;
  real<lower=0> jitter;
}

parameters {
  real mu;
  vector[N] z_customer;
  real<lower=1e-6, upper=5> sigma_customer;
  vector[T_total] eta_time;
  real<lower=1e-6, upper=5> sigma_gp;
  real<lower=1e-3, upper=52> rho;
}

transformed parameters {
  matrix[T_total, T_total] K;
  matrix[T_total, T_total] L_K;

  for (i in 1:T_total) {
    for (j in 1:T_total) {
      K[i, j] = square(sigma_gp) * exp(-0.5 * square((i - j) / rho));
    }
    K[i, i] = K[i, i] + jitter;
  }
  L_K = cholesky_decompose(K);
}

model {
  mu ~ normal(-3, 2);
  sigma_customer ~ normal(0, 1);
  z_customer ~ normal(0, 1);
  sigma_gp ~ normal(0, 1);
  rho ~ lognormal(log(8), 0.5);
  eta_time ~ multi_normal_cholesky(rep_vector(0, T_total), L_K);

  for (n in 1:N) {
    for (t in 1:T_calib) {
      x[n, t] ~ poisson_log(mu + sigma_customer * z_customer[n] + eta_time[t]);
    }
  }
}

generated quantities {
  vector[N] pred_freq_expected;

  for (n in 1:N) {
    pred_freq_expected[n] = 0;
    for (t in (T_calib + 1):T_total) {
      pred_freq_expected[n] += exp(mu + sigma_customer * z_customer[n] + eta_time[t]);
    }
  }
}
