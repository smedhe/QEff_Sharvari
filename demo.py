# Normalizes a dynamo-captured fx.GraphModule to standard aten & prims IR

def _normalize_graph_module(

    model: fx.GraphModule, example_inputs: List[Tensor], trace: List[str]

) -> fx.GraphModule:

    # model received by dynamo backends are of type _LazyGraphModule, a subclass of GraphModule.

    # make_fx is unable to trace the altered forward method of _LazyGraphModule.

    # hence, creating a standard GraphModule which can be correctly traced.

    model = fx.GraphModule(model, model.graph)
 
    # normalize captured graph to aten+prims IR upto pre-dispatch stage without decompositions

    from torch._subclasses import FakeTensorMode

    from torch._dynamo.utils import detect_fake_mode

    from torch._dispatch.python import enable_python_dispatcher

    from torch.fx.experimental.proxy_tensor import make_fx

    from torch.fx.traceback import preserve_node_meta
 
    def model_with_interpreter(*args):

        with preserve_node_meta():

            return torch.fx.Interpreter(model).run(*args)
 
    # torch dynamo already has an active FakeTensorMode for its internal tracing.

    # use it instead of creating a new mode because torch's tracing infra requires that

    # there is only one active FakeTensorMode at any time.

    fake_mode: FakeTensorMode = detect_fake_mode()

    assert fake_mode is not None

    assert fake_mode.allow_non_fake_inputs == False
 
    # temporarily allow non-fake inputs for our purpose.

    # All of this is to make make_fx(). Since make_fx is an experimental/internal API,

    # multiple global states need to be set correctly for it to work such as fake_mode,

    # enabling python dispatcher, etc. These invariants seem to change between torch

    # versions since its not part of backward-compat API.

    fake_mode.allow_non_fake_inputs = True

    fake_example_inputs = [

        fake_mode.from_tensor(t) if isinstance(t, Tensor) else t for t in example_inputs

    ]

    with enable_python_dispatcher(), fake_mode:

        model = make_fx(model_with_interpreter, tracing_mode="real", pre_dispatch=True)(

            *fake_example_inputs

        )

    fake_mode.allow_non_fake_inputs = False
 
    _qaic_print_debug(8, f"FX graph captured by _normalize_graph_module {trace = }:")

    _qaic_callif_debug(8, model.graph.print_tabular)

    _qaic_print_debug(8)
 
    return model    

