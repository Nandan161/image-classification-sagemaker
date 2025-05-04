# debug_utils.py

from smdebug.core.modes import ModeKeys

def extract_tensor_values(trial, tensor_name, mode):
    tensor = trial.tensor(tensor_name)
    steps = tensor.steps(mode=mode)
    values = [tensor.value(step, mode=mode) for step in steps]
    return steps, values

def plot_tensor_comparison(trial, tensor_name, plt):
    train_steps, train_vals = extract_tensor_values(trial, tensor_name, ModeKeys.TRAIN)
    eval_steps, eval_vals = extract_tensor_values(trial, tensor_name, ModeKeys.EVAL)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twiny()

    l1, = ax1.plot(train_steps, train_vals, label='Train', color='blue')
    l2, = ax2.plot(eval_steps, eval_vals, label='Eval', color='orange')

    ax1.set_xlabel("Training Steps", color=l1.get_color())
    ax2.set_xlabel("Eval Steps", color=l2.get_color())
    ax1.set_ylabel(tensor_name)

    plt.title(f"Tensor Visualization: {tensor_name}")
    plt.legend(handles=[l1, l2])
    plt.show()
