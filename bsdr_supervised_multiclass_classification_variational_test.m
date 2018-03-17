function prediction = bsdr_supervised_multiclass_classification_variational_test(X, state)
    rand('state', state.parameters.seed); %#ok<RAND>
    randn('state', state.parameters.seed); %#ok<RAND>

    R = size(state.bW.mu, 1) - 1;
    N = size(X, 2);
    K = size(state.bW.mu, 2);

    prediction.Z.mu = zeros(R, N);
    prediction.Z.sigma = zeros(R, N);
    for s = 1:R
        prediction.Z.mu(s, :) = state.Q.mu(:, s)' * X;
        prediction.Z.sigma(s, :) = state.parameters.sigma_z^2 + diag(X' * state.Q.sigma(:, :, s) * X);
    end

    T.mu = zeros(K, N);
    T.sigma = zeros(K, N);
    for c = 1:K
        T.mu(c, :) = state.bW.mu(:, c)' * [ones(1, N); prediction.Z.mu];
        T.sigma(c, :) = 1 + diag([ones(1, N); prediction.Z.mu]' * state.bW.sigma(:, :, c) * [ones(1, N); prediction.Z.mu]);
    end

    prediction.P = zeros(K, N);
    u = randn(1, 1, state.parameters.sample);
    for c = 1:K
        A = repmat(u, [K, N, 1]) .* repmat(T.sigma(c, :), [K, 1, state.parameters.sample]) + repmat(T.mu(c, :), [K, 1, state.parameters.sample]) - repmat(T.mu, [1, 1, state.parameters.sample]);
        A = A ./ repmat(T.sigma, [1, 1, state.parameters.sample]);
        A(c, :, :) = [];
        prediction.P(c, :) = mean(prod(normcdf(A), 3), 1);
    end
    prediction.P = prediction.P ./ repmat(sum(prediction.P, 1), K, 1);
end
