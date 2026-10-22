from copy import deepcopy
from dataclasses import dataclass
from itertools import product
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from .config import TrainingConfig
from .evaluation import evaluate_classifier
from .models import make_svm, ConfusionExperts


@dataclass
class TrainingResult:
    models: dict
    champion: str
    trials: list[dict]
    histories: dict[str, list[dict]]
    validation: dict


def _score(y, prediction):
    return float(f1_score(y, prediction, average="macro", zero_division=0))


def train_ocr(X_train, y_train, X_validation, y_validation,
              config: TrainingConfig = TrainingConfig(), checkpoint=None) -> TrainingResult:
    """Tune on validation only. No test data accepted here.

    SVM uses a convergent solver, so epoch early stopping does not apply.
    AdaBoost grows one round at a time and actually stops after patience rounds;
    restore the best estimator, rather than the last round.
    checkpoint(name, estimator, metadata) is called only for improvements.
    """
    if config.patience < 1 or config.ada_max_estimators < 1:
        raise ValueError("patience and estimator budget must be positive")
    y_train, y_validation = np.asarray(y_train), np.asarray(y_validation)
    if set(y_train) != set(y_validation):
        raise ValueError("Training and validation must contain the same classes")
    trials, models, histories, best_svm = [], {}, {}, (-np.inf, None)
    for c, kernel in product(config.svm_c, config.svm_kernels):
        model = make_svm(c, kernel, config.seed, probability=False).fit(X_train, y_train)
        score = _score(y_validation, model.predict(X_validation))
        params = dict(C=c, kernel=kernel)
        trials.append(dict(family="svm", params=params, macro_f1=score))
        if score > best_svm[0]:
            best_svm = score, params
    if best_svm[1] is None:
        raise ValueError("Empty SVM search space")
    best_params = best_svm[1]
    models["svm"] = make_svm(best_params["C"], best_params["kernel"], config.seed).fit(X_train, y_train)
    if checkpoint:
        checkpoint("svm", models["svm"], dict(params=best_params, macro_f1=best_svm[0]))

    best_ada = -np.inf
    for depth, rate in product(config.ada_depths, config.ada_learning_rates):
        key = f"adaboost_depth{depth}_lr{rate}"
        history, best_trial, best_model, stale = [], -np.inf, None, 0
        # AdaBoost has no warm_start. Implement the SAMME round recurrence using
        # sklearn's public one-estimator AdaBoost fit, carrying sample weights.
        weights = np.full(len(y_train), 1.0 / len(y_train))
        ensemble = None
        stop_reason = "estimator_budget"
        for stage in range(1, config.ada_max_estimators + 1):
            one = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=depth, min_samples_leaf=2),
                                     n_estimators=1, learning_rate=rate, random_state=config.seed + stage)
            try:
                one.fit(X_train, y_train, sample_weight=weights)
            except ValueError as error:
                if "worse than random" in str(error) and ensemble is not None:
                    stop_reason = "weak_learner_worse_than_random"
                    break
                raise
            learner, alpha, error = one.estimators_[0], float(one.estimator_weights_[0]), float(one.estimator_errors_[0])
            if ensemble is None:
                ensemble = deepcopy(one)
            else:
                ensemble.estimators_.append(learner)
                ensemble.estimator_weights_ = np.append(ensemble.estimator_weights_, alpha)
                ensemble.estimator_errors_ = np.append(ensemble.estimator_errors_, error)
            ensemble.n_estimators = len(ensemble.estimators_)
            prediction = ensemble.predict(X_validation)
            score = _score(y_validation, prediction)
            entry = dict(stage=stage, train_accuracy=float(accuracy_score(y_train, ensemble.predict(X_train))),
                         validation_macro_f1=score, weighted_error=error)
            history.append(entry)
            if score > best_trial + config.min_delta:
                best_trial, best_model, stale = score, deepcopy(ensemble), 0
            else:
                stale += 1
            if score > best_ada + config.min_delta:
                best_ada = score
                models["adaboost"] = deepcopy(ensemble)
                if checkpoint:
                    checkpoint("adaboost", ensemble, dict(depth=depth, learning_rate=rate, **entry))
            if stale >= config.patience or error <= 0:
                stop_reason = "perfect_training_fit" if error <= 0 else "validation_patience"
                break
            incorrect = learner.predict(X_train) != y_train
            # SAMME update; subtract max log weight to avoid overflow.
            log_weights = np.log(np.maximum(weights, np.finfo(float).tiny)) + alpha * incorrect
            weights = np.exp(log_weights - log_weights.max())
            weights /= weights.sum()
        histories[key] = history
        trials.append(dict(family="adaboost", params=dict(depth=depth, learning_rate=rate),
                           macro_f1=best_trial, best_stage=len(best_model.estimators_) if best_model else 0,
                           stopped_stage=len(history), stop_reason=stop_reason))

    experts = ConfusionExperts(c=best_params["C"], kernel=best_params["kernel"],
                               groups=config.expert_groups, seed=config.seed).fit(X_train, y_train)
    best_strength, expert_score = 0.0, -np.inf
    for strength in (0.0, 0.3, 0.6, 1.0):
        experts.strength = strength
        score = _score(y_validation, experts.predict(X_validation))
        trials.append(dict(family="confusion_experts", params=dict(strength=strength), macro_f1=score))
        if score > expert_score:
            best_strength, expert_score = strength, score
    experts.strength = best_strength
    models["confusion_experts"] = experts
    if checkpoint:
        checkpoint("confusion_experts", experts, dict(strength=best_strength, macro_f1=expert_score))
    validation = {name: evaluate_classifier(model, X_validation, y_validation) for name, model in models.items()}
    champion = max(validation, key=lambda name: validation[name]["macro_f1"])
    return TrainingResult(models, champion, trials, histories, validation)
