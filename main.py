from jinja2 import Environment, FileSystemLoader

env = Environment(loader=FileSystemLoader('template'))
template = env.get_template('index.html.j2')
rendered_html = template.render()  # Add variables if needed

with open('index.html', 'w', encoding='utf-8') as f:
    f.write(rendered_html)