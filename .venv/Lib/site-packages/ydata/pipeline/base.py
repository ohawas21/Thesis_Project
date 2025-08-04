"""
    Defines the base Class for YData's Pipeline's
"""
import timeit
from contextlib import contextmanager

from sklearn.base import clone
from sklearn.pipeline import Pipeline as skPipeline
from sklearn.pipeline import _fit_transform_one
from sklearn.utils.validation import check_memory

def _message_with_time(source, message, time):
    """Create one line message for logging purposes.

    Parameters
    ----------
    source : str
        String indicating the source or the reference of the message.

    message : str
        Short message.

    time : int
        Time in seconds.
    """
    start_message = "[%s] " % source

    # adapted from joblib.logger.short_format_time without the Windows -.1s
    # adjustment
    if time > 60:
        time_str = "%4.1fmin" % (time / 60)
    else:
        time_str = " %5.1fs" % time
    end_message = " %s, total=%s" % (message, time_str)
    dots_len = 70 - len(start_message) - len(end_message)
    return "%s%s%s" % (start_message, dots_len * ".", end_message)

@contextmanager
def _print_elapsed_time(source, message=None):
    """Log elapsed time to stdout when the context is exited.

    Parameters
    ----------
    source : str
        String indicating the source or the reference of the message.

    message : str, default=None
        Short message. If None, nothing will be printed.

    Returns
    -------
    context_manager
        Prints elapsed time upon exit if verbose.
    """
    if message is None:
        yield
    else:
        start = timeit.default_timer()
        yield
        print(_message_with_time(source, message, timeit.default_timer() - start))

def _fit_transform_one(  # noqa: F811
    transformer, X, input_dtypes, weight, message_clsname="", message=None, **fit_params
):
    """Fits ``transformer`` to ``X`` and ``y``.

    The transformed result is returned with the fitted transformer. If
    ``weight`` is not ``None``, the result will be multiplied by
    ``weight``.
    """
    if hasattr(transformer, "fit_transform"):
        res = transformer.fit_transform(
            X, input_dtypes=input_dtypes, **fit_params)
    else:
        res = transformer.fit(
            X, input_dtypes=input_dtypes, **fit_params).transform(X)

    if weight is None:
        return res, transformer


class Pipeline(skPipeline):
    def __init__(self, steps):
        super().__init__(steps)

    def _check_fit_params(self, **fit_params):
        fit_params_steps = {name: {} for name, step in self.steps if step is not None}
        for pname, pval in fit_params.items():
            if "__" not in pname:
                raise ValueError(
                    "Pipeline.fit does not accept the {} parameter. "
                    "You can pass parameters to specific steps of your "
                    "pipeline using the stepname__parameter format, e.g. "
                    "`Pipeline.fit(X, y, logisticregression__sample_weight"
                    "=sample_weight)`.".format(pname)
                )
            step, param = pname.split("__", 1)
            fit_params_steps[step][param] = pval
        return fit_params_steps

    def _fit(self, X, input_dtypes, **fit_params_steps):
        # shallow copy of steps - this should really be steps_
        self.steps = list(self.steps)
        self._validate_steps()
        # Setup the memory
        memory = check_memory(self.memory)

        fit_transform_one_cached = memory.cache(_fit_transform_one)
        output_dtypes = input_dtypes
        for step_idx, name, transformer in self._iter(
            with_final=False, filter_passthrough=False
        ):
            if transformer is None or transformer == "passthrough":
                with _print_elapsed_time("Pipeline", self._log_message(step_idx)):
                    continue

            if hasattr(memory, "location") and memory.location is None:
                # we do not clone when caching is disabled to
                # preserve backward compatibility
                cloned_transformer = transformer
            else:
                cloned_transformer = clone(transformer)
            # Fit or load from cache the current transformer
            X, fitted_transformer = fit_transform_one_cached(
                cloned_transformer,
                X,
                output_dtypes,
                None,
                message_clsname="Pipeline",
                message=self._log_message(step_idx),
                **fit_params_steps[name],
            )
            # Replace the transformer of the step with the fitted
            # transformer. This is necessary when loading the transformer
            # from the cache.
            self.steps[step_idx] = (name, fitted_transformer)
            output_dtypes = fitted_transformer.output_dtypes
        return X, output_dtypes

    def fit(self, X, input_dtypes, **fit_params):
        fit_params_steps = self._check_fit_params(**fit_params)
        Xt, output_types = self._fit(
            X, input_dtypes=input_dtypes, **fit_params_steps)
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if self._final_estimator != "passthrough":
                fit_params_last_step = fit_params_steps[self.steps[-1][0]]
                self._final_estimator.fit(
                    X=Xt, input_dtypes=output_types, **fit_params_last_step
                )
        return self

    def fit_transform(self, X, input_dtypes, **fit_params):
        fit_params_steps = self._check_fit_params(**fit_params)
        Xt, output_types = self._fit(X, input_dtypes, **fit_params_steps)

        last_step = self._final_estimator
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if last_step == "passthrough":
                return Xt
            fit_params_last_step = fit_params_steps[self.steps[-1][0]]
            input_dtypes = output_types
            if hasattr(last_step, "fit_transform"):
                return last_step.fit_transform(
                    X=Xt, input_dtypes=input_dtypes, **fit_params_last_step
                )
            else:
                return last_step.fit(
                    X=Xt, input_dtypes=input_dtypes, **fit_params_last_step
                ).transform(X=Xt)
