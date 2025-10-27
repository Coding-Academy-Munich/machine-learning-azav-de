import html
from IPython.core.getipython import get_ipython
from IPython.display import display, HTML
from IPython.core.magic import register_line_magic

def oneline(*args, titles=None):
    """
    Displays multiple Python objects side-by-side in a Jupyter notebook,
    optionally with titles.

    Args:
        *args: The Python objects to display (e.g., arrays, DataFrames, strings).
        titles (list, optional): A list of strings to use as titles for each object.
                                 Must be the same length as args.
    """

    # Handle titles
    if titles is None:
        titles = [''] * len(args)
    elif len(titles) != len(args):
        print("Warning: Number of titles does not match number of objects.")
        titles = [''] * len(args) # Reset to avoid errors

    html_content = ""
    for i, (obj, title) in enumerate(zip(args, titles)):

        # Get the HTML representation of the object
        if hasattr(obj, '_repr_html_'):
            # This works for DataFrames, images, etc.
            repr_html = obj._repr_html_()
        else:
            # This works for NumPy arrays, lists, strings, etc.
            repr_html = f"<pre>{html.escape(repr(obj))}</pre>"

        # Add title if provided
        title_html = f'<b>{html.escape(title)}</b>' if title else ''

        # Wrap in a div
        html_content += f'''
            <div style="margin-right: 25px; vertical-align: top; text-align: left;">
                {title_html}
                {repr_html}
            </div>
        '''

    # Wrap all content in a single flex container
    final_html = f'<div style="display: flex; justify-content: left;">{html_content}</div>'
    display(HTML(final_html))

    from IPython.core.magic import register_line_magic


from IPython.core.magic import register_line_magic

@register_line_magic
def side_by_side(line):
    """
    Line magic to display variables and expressions side-by-side.
    Usage: %side_by_side A, A.shape, A + B
    Alias: %sbs

    Titles will be the expressions themselves.
    WARNING: Does not support commas within an expression (e.g., a tuple).
    """
    # Get the expression strings by splitting the input line
    expr_strings = [s.strip() for s in line.split(',')]

    # Get the IPython shell instance
    ipy = get_ipython()

    objs_to_display = []

    for expr in expr_strings:
        if not expr: # Skip empty strings if user adds a trailing comma
            continue
        try:
            # Evaluate the expression in the user's namespace
            obj = ipy.ev(expr)
            objs_to_display.append(obj)
        except Exception as e:
            print(f"Error evaluating expression '{expr}': {e}")
            return # Stop if any expression fails

    # Call our helper function using the expressions as titles
    oneline(*objs_to_display, titles=expr_strings)

# Register the alias
@register_line_magic
def sbs(line):
    """
    Alias for side_by_side magic.
    Usage: %sbs A, A.shape, A + B
    """
    return side_by_side(line)

@register_line_magic
def with_shape(line):
    """
    Line magic to display variables/expressions along with their shapes.
    Usage: %with_shape A, B, A + B
    Alias: %ws

    For each expression e_i, displays: e_i, e_i.shape (or (e_i).shape for complex expressions)
    """
    # Get the expression strings by splitting the input line
    expr_strings = [s.strip() for s in line.split(',')]

    # Get the IPython shell instance
    ipy = get_ipython()

    objs_to_display = []
    titles_to_display = []

    for expr in expr_strings:
        if not expr:  # Skip empty strings if user adds a trailing comma
            continue
        try:
            # Evaluate the expression in the user's namespace
            obj = ipy.ev(expr)

            # Add the object itself
            objs_to_display.append(obj)
            titles_to_display.append(expr)

            # Add the shape if the object has a shape attribute
            if hasattr(obj, 'shape'):
                objs_to_display.append(obj.shape)

                # Determine if we need parentheses around the expression
                # Simple heuristic: if the expression contains operators or spaces, add parens
                needs_parens = any(op in expr for op in ['+', '-', '*', '/', '@', ' ', '(', '['])

                if needs_parens:
                    shape_title = f"({expr}).shape"
                else:
                    shape_title = f"{expr}.shape"

                titles_to_display.append(shape_title)

        except Exception as e:
            print(f"Error evaluating expression '{expr}': {e}")
            return  # Stop if any expression fails

    # Call our helper function with the collected objects and titles
    oneline(*objs_to_display, titles=titles_to_display)

# Register the alias
@register_line_magic
def ws(line):
    """
    Alias for with_shape magic.
    Usage: %ws A, B, A + B
    """
    return with_shape(line)