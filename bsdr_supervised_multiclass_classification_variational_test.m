% Mehmet Gonen (mehmet.gonen@gmail.com)

function prediction = bsdr_supervised_multiclass_classification_variational_test(X, state)
    rand('state', state.parameters.seed); %#ok<RAND>
    randn('state', state.parameters.seed); %#ok<RAND>

    R = size(state.bW.mean, 1) - 1;
    N = size(X, 2);
    K = size(state.bW.mean, 2);

    prediction.Z.mean = zeros(R, N);
    prediction.Z.covariance = zeros(R, N);
    for s = 1:R
        prediction.Z.mean(s, :) = state.Q.mean(:, s)' * X;
        prediction.Z.covariance(s, :) = state.parameters.sigmaz^2 + diag(X' * state.Q.covariance(:, :, s) * X);
    end

    T.mean = zeros(K, N);
    T.covariance = zeros(K, N);
    for c = 1:K
        T.mean(c, :) = state.bW.mean(:, c)' * [ones(1, N); prediction.Z.mean];
        T.covariance(c, :) = 1 + diag([ones(1, N); prediction.Z.mean]' * state.bW.covariance(:, :, c) * [ones(1, N); prediction.Z.mean]);
    end

    prediction.P = zeros(K, N);
    u = randn(1, 1, state.parameters.sample);
    for c = 1:K
        A = repmat(u, [K, N, 1]) .* repmat(T.covariance(c, :), [K, 1, state.parameters.sample]) + repmat(T.mean(c, :), [K, 1, state.parameters.sample]) - repmat(T.mean, [1, 1, state.parameters.sample]);
        A = A ./ repmat(T.covariance, [1, 1, state.parameters.sample]);
        A(c, :, :) = [];
        prediction.P(c, :) = mean(prod(normcdf(A), 3), 1);
    end
    prediction.P = prediction.P ./ repmat(sum(prediction.P, 1), K, 1);
end