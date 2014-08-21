% Mehmet Gonen (mehmet.gonen@gmail.com)

function state = bsdr_supervised_multiclass_classification_variational_train(X, y, parameters)
    rand('state', parameters.seed); %#ok<RAND>
    randn('state', parameters.seed); %#ok<RAND>

    D = size(X, 1);
    N = size(X, 2);
    K = max(y);
    R = parameters.R;
    sigmaz = parameters.sigmaz;

    log2pi = log(2 * pi);

    switch parameters.prior_phi
        case 'ard'
            phi.shape = (parameters.alpha_phi + 0.5 * D) * ones(R, 1);
            phi.scale = parameters.beta_phi * ones(R, 1);
        otherwise
            Phi.shape = (parameters.alpha_phi + 0.5) * ones(D, R);
            Phi.scale = parameters.beta_phi * ones(D, R);
    end
    Q.mean = randn(D, R);
    Q.covariance = repmat(eye(D, D), [1, 1, R]);
    Z.mean = randn(R, N);
    Z.covariance = eye(R, R);
    lambda.shape = (parameters.alpha_lambda + 0.5) * ones(K, 1);
    lambda.scale = parameters.beta_lambda * ones(K, 1);
    Psi.shape = (parameters.alpha_psi + 0.5) * ones(R, K);
    Psi.scale = parameters.beta_psi * ones(R, K);
    bW.mean = randn(R + 1, K);
    bW.covariance = repmat(eye(R + 1, R + 1), [1, 1, K]);
    T.mean = zeros(K, N);
    T.covariance = eye(K, K);
    for i = 1:N
        while 1
            T.mean(:, i) = randn(K, 1);
            if T.mean(y(i), i) == max(T.mean(:, i))
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
                    phi.scale(s) = 1 / (1 / parameters.beta_phi + 0.5 * (Q.mean(:, s)' * Q.mean(:, s) + sum(diag(Q.covariance(:, :, s)))));
                end
                %%%% update Q
                for s = 1:R
                    Q.covariance(:, :, s) = (phi.shape(s) * phi.scale(s) * eye(D, D) + XXT / sigmaz^2) \ eye(D, D);
                    Q.mean(:, s) = Q.covariance(:, :, s) * (X * Z.mean(s, :)' / sigmaz^2);
                end
            otherwise
                %%%% update Phi
                Phi.scale = 1 ./ (1 / parameters.beta_phi + 0.5 * (Q.mean.^2 + reshape(Q.covariance(phi_indices), D, R)));
                %%%% update Q
                for s = 1:R
                    Q.covariance(:, :, s) = (diag(Phi.shape(:, s) .* Phi.scale(:, s)) + XXT / sigmaz^2) \ eye(D, D);
                    Q.mean(:, s) = Q.covariance(:, :, s) * (X * Z.mean(s, :)' / sigmaz^2);
                end
        end
        %%%% update Z
        Z.covariance = (eye(R, R) / sigmaz^2 + bW.mean(2:R + 1, :) * bW.mean(2:R + 1, :)' + sum(bW.covariance(2:R + 1, 2:R + 1, :), 3)) \ eye(R, R);
        Z.mean = Z.covariance * (Q.mean' * X / sigmaz^2 + bW.mean(2:end, :) * T.mean - repmat(bW.mean(2:R + 1, :) * bW.mean(1, :)' + sum(bW.covariance(1, 2:R + 1, :), 3)', 1, N));
        %%%% update lambda
        lambda.scale = 1 ./ (1 / parameters.beta_lambda + 0.5 * (bW.mean(1, :)'.^2 + squeeze(bW.covariance(1, 1, :))));
        %%%% update Psi
        Psi.scale = 1 ./ (1 / parameters.beta_psi + 0.5 * (bW.mean(2:R + 1, :).^2 + reshape(bW.covariance(psi_indices), R, K)));
        %%%% update b and W
        for c = 1:K
            bW.covariance(:, :, c) = [lambda.shape(c, 1) * lambda.scale(c, 1) + N, sum(Z.mean, 2)'; sum(Z.mean, 2), diag(Psi.shape(:, c) .* Psi.scale(:, c)) + Z.mean * Z.mean' + N * Z.covariance] \ eye(R + 1, R + 1);
            bW.mean(:, c) = bW.covariance(:, :, c) * [ones(1, N); Z.mean] * T.mean(c, :)';
        end
        %%%% update T
        T.mean = bW.mean(2:R + 1, :)' * Z.mean + repmat(bW.mean(1, :)', 1, N);
        for c = 1:K
            pos = find(y == c);
            [normalization(pos, 1), T.mean(:, pos)] = truncated_normal_mean(T.mean(:, pos), c, parameters.sample, 0);
        end

        lb = 0;
        switch parameters.prior_phi
            case 'ard'
                %%%% p(phi)
                lb = lb + sum((parameters.alpha_phi - 1) * (psi(phi.shape) + log(phi.scale)) - phi.shape .* phi.scale / parameters.beta_phi - gammaln(parameters.alpha_phi) - parameters.alpha_phi * log(parameters.beta_phi));
                %%%% p(Q | phi)
                for s = 1:R
                    lb = lb - 0.5 * Q.mean(:, s)' * (phi.shape(s) * phi.scale(s) * eye(D, D)) * Q.mean(:, s) - 0.5 * (D * log2pi - D * log(phi.shape(s) * phi.scale(s)));
                end
            otherwise
                %%%% p(Phi)
                lb = lb + sum(sum((parameters.alpha_phi - 1) * (psi(Phi.shape) + log(Phi.scale)) - Phi.shape .* Phi.scale / parameters.beta_phi - gammaln(parameters.alpha_phi) - parameters.alpha_phi * log(parameters.beta_phi)));
                %%%% p(Q | Phi)
                for s = 1:R
                    lb = lb - 0.5 * Q.mean(:, s)' * diag(Phi.shape(:, s) .* Phi.scale(:, s)) * Q.mean(:, s) - 0.5 * (D * log2pi - sum(log(Phi.shape(:, s) .* Phi.scale(:, s))));
                end
        end
        %%%% p(Z | Q, X)
        lb = lb - 0.5 * (sum(sum(Z.mean .* Z.mean)) + N * sum(diag(Z.covariance))) + sum(sum((Q.mean' * X) .* Z.mean)) - 0.5 * sum(sum(X .* ((Q.mean * Q.mean' + sum(Q.covariance, 3)) * X))) - 0.5 * N * D * (log2pi + 2 * log(sigmaz));
        %%%% p(lambda)
        lb = lb + sum((parameters.alpha_lambda - 1) * (psi(lambda.shape) + log(lambda.scale)) - lambda.shape .* lambda.scale / parameters.beta_lambda - gammaln(parameters.alpha_lambda) - parameters.alpha_lambda * log(parameters.beta_lambda));        
        %%%% p(b | lambda)
        lb = lb - 0.5 * bW.mean(1, :) * diag(lambda.shape(:, 1) .* lambda.scale(:, 1)) * bW.mean(1, :)' - 0.5 * (K * log2pi - sum(log(lambda.shape(:, 1) .* lambda.scale(:, 1))));
        %%%% p(Psi)
        lb = lb + sum(sum((parameters.alpha_psi - 1) * (psi(Psi.shape) + log(Psi.scale)) - Psi.shape .* Psi.scale / parameters.beta_psi - gammaln(parameters.alpha_psi) - parameters.alpha_psi * log(parameters.beta_psi)));
        %%%% p(W | Psi)
        for c = 1:K
            lb = lb - 0.5 * bW.mean(2:R + 1, c)' * diag(Psi.shape(:, c) .* Psi.scale(:, c)) * bW.mean(2:R + 1, c) - 0.5 * (R * log2pi - sum(log(Psi.shape(:, c) .* Psi.scale(:, c))));
        end
        %%%% p(T | b, W, Z) p(y | T)
        WWT.mean = bW.mean(2:R + 1, :) * bW.mean(2:R + 1, :)' + sum(bW.covariance(2:R + 1, 2:R + 1, :), 3);
        lb = lb - 0.5 * (sum(sum(T.mean .* T.mean)) + N * K) + sum(bW.mean(1, :) * T.mean) + sum(sum(Z.mean .* (bW.mean(2:R + 1, :) * T.mean))) - 0.5 * (N * trace(WWT.mean * Z.covariance) + sum(sum(Z.mean .* (WWT.mean * Z.mean)))) - 0.5 * N * (bW.mean(1, :) * bW.mean(1, :)' + sum(bW.covariance(1, 1, :))) - sum(Z.mean' * (bW.mean(2:R + 1, :) * bW.mean(1, :)' + sum(bW.covariance(2:R + 1, 1, :), 3))) - 0.5 * N * K * log2pi;

        switch parameters.prior_phi
            case 'ard'
                %%%% q(phi)
                lb = lb + sum(phi.shape + log(phi.scale) + gammaln(phi.shape) + (1 - phi.shape) .* psi(phi.shape));
            otherwise
                %%%% q(Phi)
                lb = lb + sum(sum(Phi.shape + log(Phi.scale) + gammaln(Phi.shape) + (1 - Phi.shape) .* psi(Phi.shape)));
        end
        %%%% q(Q)
        for s = 1:R
            lb = lb + 0.5 * (D * (log2pi + 1) + logdet(Q.covariance(:, :, s)));
        end
        %%%% q(Z)
        lb = lb + 0.5 * N * (R * (log2pi + 1) + logdet(Z.covariance));
        %%%% q(lambda)
        lb = lb + sum(lambda.shape + log(lambda.scale) + gammaln(lambda.shape) + (1 - lambda.shape) .* psi(lambda.shape));
        %%%% q(Psi)
        lb = lb + sum(sum(Psi.shape + log(Psi.scale) + gammaln(Psi.shape) + (1 - Psi.shape) .* psi(Psi.shape)));
        %%%% q(b, W)
        for c = 1:K
            lb = lb + 0.5 * ((R + 1) * (log2pi + 1) + logdet(bW.covariance(:, :, c))); 
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