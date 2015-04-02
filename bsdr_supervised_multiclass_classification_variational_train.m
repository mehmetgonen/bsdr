% Mehmet Gonen (mehmet.gonen@gmail.com)

function state = bsdr_supervised_multiclass_classification_variational_train(X, y, parameters)
    rand('state', parameters.seed); %#ok<RAND>
    randn('state', parameters.seed); %#ok<RAND>

    D = size(X, 1);
    N = size(X, 2);
    K = max(y);
    R = parameters.R;
    sigma_z = parameters.sigma_z;

    log2pi = log(2 * pi);

    switch parameters.prior_phi
        case 'ard'
            phi.alpha = (parameters.alpha_phi + 0.5 * D) * ones(R, 1);
            phi.beta = parameters.beta_phi * ones(R, 1);
        otherwise
            Phi.alpha = (parameters.alpha_phi + 0.5) * ones(D, R);
            Phi.beta = parameters.beta_phi * ones(D, R);
    end
    Q.mu = randn(D, R);
    Q.sigma = repmat(eye(D, D), [1, 1, R]);
    Z.mu = randn(R, N);
    Z.sigma = eye(R, R);
    lambda.alpha = (parameters.alpha_lambda + 0.5) * ones(K, 1);
    lambda.beta = parameters.beta_lambda * ones(K, 1);
    Psi.alpha = (parameters.alpha_psi + 0.5) * ones(R, K);
    Psi.beta = parameters.beta_psi * ones(R, K);
    bW.mu = randn(R + 1, K);
    bW.sigma = repmat(eye(R + 1, R + 1), [1, 1, K]);
    T.mu = zeros(K, N);
    T.sigma = eye(K, K);
    for i = 1:N
        while 1
            T.mu(:, i) = randn(K, 1);
            if T.mu(y(i), i) == max(T.mu(:, i))
                break;
            end
        end
    end
    normalization = zeros(N, 1);

    XXT = X * X';
    phi_indices = repmat(logical(eye(D, D)), [1, 1, R]);
    psi_indices = repmat(logical([zeros(1, R + 1); zeros(R, 1), eye(R, R)]), [1, 1, K]);

    if parameters.progress == 1
        bounds = zeros(parameters.iteration, 1);
    end

    for iter = 1:parameters.iteration
        if mod(iter, 1) == 0
            fprintf(1, '.');
        end
        if mod(iter, 10) == 0
            fprintf(1, ' %5d\n', iter);
        end
        
        switch parameters.prior_phi
            case 'ard'
                %%%% update phi
                for s = 1:R
                    phi.beta(s) = 1 / (1 / parameters.beta_phi + 0.5 * (Q.mu(:, s)' * Q.mu(:, s) + sum(diag(Q.sigma(:, :, s)))));
                end
                %%%% update Q
                for s = 1:R
                    Q.sigma(:, :, s) = (phi.alpha(s) * phi.beta(s) * eye(D, D) + XXT / sigma_z^2) \ eye(D, D);
                    Q.mu(:, s) = Q.sigma(:, :, s) * (X * Z.mu(s, :)' / sigma_z^2);
                end
            otherwise
                %%%% update Phi
                Phi.beta = 1 ./ (1 / parameters.beta_phi + 0.5 * (Q.mu.^2 + reshape(Q.sigma(phi_indices), D, R)));
                %%%% update Q
                for s = 1:R
                    Q.sigma(:, :, s) = (diag(Phi.alpha(:, s) .* Phi.beta(:, s)) + XXT / sigma_z^2) \ eye(D, D);
                    Q.mu(:, s) = Q.sigma(:, :, s) * (X * Z.mu(s, :)' / sigma_z^2);
                end
        end
        %%%% update Z
        Z.sigma = (eye(R, R) / sigma_z^2 + bW.mu(2:R + 1, :) * bW.mu(2:R + 1, :)' + sum(bW.sigma(2:R + 1, 2:R + 1, :), 3)) \ eye(R, R);
        Z.mu = Z.sigma * (Q.mu' * X / sigma_z^2 + bW.mu(2:end, :) * T.mu - repmat(bW.mu(2:R + 1, :) * bW.mu(1, :)' + sum(bW.sigma(1, 2:R + 1, :), 3)', 1, N));
        %%%% update lambda
        lambda.beta = 1 ./ (1 / parameters.beta_lambda + 0.5 * (bW.mu(1, :)'.^2 + squeeze(bW.sigma(1, 1, :))));
        %%%% update Psi
        Psi.beta = 1 ./ (1 / parameters.beta_psi + 0.5 * (bW.mu(2:R + 1, :).^2 + reshape(bW.sigma(psi_indices), R, K)));
        %%%% update b and W
        for c = 1:K
            bW.sigma(:, :, c) = [lambda.alpha(c, 1) * lambda.beta(c, 1) + N, sum(Z.mu, 2)'; sum(Z.mu, 2), diag(Psi.alpha(:, c) .* Psi.beta(:, c)) + Z.mu * Z.mu' + N * Z.sigma] \ eye(R + 1, R + 1);
            bW.mu(:, c) = bW.sigma(:, :, c) * [ones(1, N); Z.mu] * T.mu(c, :)';
        end
        %%%% update T
        T.mu = bW.mu(2:R + 1, :)' * Z.mu + repmat(bW.mu(1, :)', 1, N);
        for c = 1:K
            pos = find(y == c);
            [normalization(pos, 1), T.mu(:, pos)] = truncated_normal_mean(T.mu(:, pos), c, parameters.sample, 0);
        end

        lb = 0;
        switch parameters.prior_phi
            case 'ard'
                %%%% p(phi)
                lb = lb + sum((parameters.alpha_phi - 1) * (psi(phi.alpha) + log(phi.beta)) - phi.alpha .* phi.beta / parameters.beta_phi - gammaln(parameters.alpha_phi) - parameters.alpha_phi * log(parameters.beta_phi));
                %%%% p(Q | phi)
                for s = 1:R
                    lb = lb - 0.5 * Q.mu(:, s)' * (phi.alpha(s) * phi.beta(s) * eye(D, D)) * Q.mu(:, s) - 0.5 * (D * log2pi - D * (psi(phi.alpha(s)) + log(phi.beta(s))));
                end
            otherwise
                %%%% p(Phi)
                lb = lb + sum(sum((parameters.alpha_phi - 1) * (psi(Phi.alpha) + log(Phi.beta)) - Phi.alpha .* Phi.beta / parameters.beta_phi - gammaln(parameters.alpha_phi) - parameters.alpha_phi * log(parameters.beta_phi)));
                %%%% p(Q | Phi)
                for s = 1:R
                    lb = lb - 0.5 * Q.mu(:, s)' * diag(Phi.alpha(:, s) .* Phi.beta(:, s)) * Q.mu(:, s) - 0.5 * (D * log2pi - sum(psi(Phi.alpha(:, s)) + log(Phi.beta(:, s))));
                end
        end
        %%%% p(Z | Q, X)
        lb = lb - 0.5 * sigma_z^-2 * (sum(sum(Z.mu .* Z.mu)) + N * sum(diag(Z.sigma))) + sigma_z^-2 * sum(sum((Q.mu' * X) .* Z.mu)) - 0.5 * sigma_z^-2 * sum(sum(X .* ((Q.mu * Q.mu' + sum(Q.sigma, 3)) * X))) - 0.5 * N * D * (log2pi + 2 * log(sigma_z));
        %%%% p(lambda)
        lb = lb + sum((parameters.alpha_lambda - 1) * (psi(lambda.alpha) + log(lambda.beta)) - lambda.alpha .* lambda.beta / parameters.beta_lambda - gammaln(parameters.alpha_lambda) - parameters.alpha_lambda * log(parameters.beta_lambda));        
        %%%% p(b | lambda)
        lb = lb - 0.5 * bW.mu(1, :) * diag(lambda.alpha(:, 1) .* lambda.beta(:, 1)) * bW.mu(1, :)' - 0.5 * (K * log2pi - sum(psi(lambda.alpha(:, 1)) + log(lambda.beta(:, 1))));
        %%%% p(Psi)
        lb = lb + sum(sum((parameters.alpha_psi - 1) * (psi(Psi.alpha) + log(Psi.beta)) - Psi.alpha .* Psi.beta / parameters.beta_psi - gammaln(parameters.alpha_psi) - parameters.alpha_psi * log(parameters.beta_psi)));
        %%%% p(W | Psi)
        for c = 1:K
            lb = lb - 0.5 * bW.mu(2:R + 1, c)' * diag(Psi.alpha(:, c) .* Psi.beta(:, c)) * bW.mu(2:R + 1, c) - 0.5 * (R * log2pi - sum(psi(Psi.alpha(:, c)) + log(Psi.beta(:, c))));
        end
        %%%% p(T | b, W, Z)
        WWT.mu = bW.mu(2:R + 1, :) * bW.mu(2:R + 1, :)' + sum(bW.sigma(2:R + 1, 2:R + 1, :), 3);
        lb = lb - 0.5 * (sum(sum(T.mu .* T.mu)) + N * K) + sum(bW.mu(1, :) * T.mu) + sum(sum(Z.mu .* (bW.mu(2:R + 1, :) * T.mu))) - 0.5 * (N * trace(WWT.mu * Z.sigma) + sum(sum(Z.mu .* (WWT.mu * Z.mu)))) - 0.5 * N * (bW.mu(1, :) * bW.mu(1, :)' + sum(bW.sigma(1, 1, :))) - sum(Z.mu' * (bW.mu(2:R + 1, :) * bW.mu(1, :)' + sum(bW.sigma(2:R + 1, 1, :), 3))) - 0.5 * N * K * log2pi;

        switch parameters.prior_phi
            case 'ard'
                %%%% q(phi)
                lb = lb + sum(phi.alpha + log(phi.beta) + gammaln(phi.alpha) + (1 - phi.alpha) .* psi(phi.alpha));
            otherwise
                %%%% q(Phi)
                lb = lb + sum(sum(Phi.alpha + log(Phi.beta) + gammaln(Phi.alpha) + (1 - Phi.alpha) .* psi(Phi.alpha)));
        end
        %%%% q(Q)
        for s = 1:R
            lb = lb + 0.5 * (D * (log2pi + 1) + logdet(Q.sigma(:, :, s)));
        end
        %%%% q(Z)
        lb = lb + 0.5 * N * (R * (log2pi + 1) + logdet(Z.sigma));
        %%%% q(lambda)
        lb = lb + sum(lambda.alpha + log(lambda.beta) + gammaln(lambda.alpha) + (1 - lambda.alpha) .* psi(lambda.alpha));
        %%%% q(Psi)
        lb = lb + sum(sum(Psi.alpha + log(Psi.beta) + gammaln(Psi.alpha) + (1 - Psi.alpha) .* psi(Psi.alpha)));
        %%%% q(b, W)
        for c = 1:K
            lb = lb + 0.5 * ((R + 1) * (log2pi + 1) + logdet(bW.sigma(:, :, c))); 
        end
        %%%% q(T)
        lb = lb + 0.5 * N * K * (log2pi + 1) + sum(log(normalization));

        bounds(iter) = lb;
    end

    switch parameters.prior_phi
        case 'ard'
            state.phi = phi;
        otherwise
            state.Phi = Phi;
    end
    state.Q = Q;
    state.lambda = lambda;
    state.Psi = Psi;
    state.bW = bW;
    if parameters.progress == 1
        state.bounds = bounds;
    end
    state.parameters = parameters;
end

function ld = logdet(Sigma)
    U = chol(Sigma);
    ld = 2 * sum(log(diag(U)));
end

function [normalization, expectation] = truncated_normal_mean(centers, active, S, tube)
    K = size(centers, 1);
    N = size(centers, 2);    
    diff = repmat(centers(active, :), K, 1) - centers - tube;
    u = randn([1, N, S]);
    q = normcdf(repmat(u, [K, 1, 1]) + repmat(diff, [1, 1, S]));
    pr = repmat(prod(q, 1), [K, 1, 1]);
    pr = pr ./ q;
    ind = [1:active - 1, active + 1:K];
    pr(ind, :, :) = pr(ind, :, :) ./ repmat(q(active, :, :), [K - 1, 1, 1]);
    pr(ind, :, :) = pr(ind, :, :) .* normpdf(repmat(u, [K - 1, 1, 1]) + repmat(diff(ind, :), [1, 1, S]));

    normalization = mean(pr(active, :, :), 3);
    expectation = zeros(K, N);
    expectation(ind, :) = centers(ind, :) - repmat(1 ./ normalization, K - 1, 1) .* reshape(mean(pr(ind, :, :), 3), K - 1, N);
    expectation(active, :) = centers(active, :) + sum(centers(ind, :) - expectation(ind, :), 1);
end