# This is a good start
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.label import Label
# from kivy.animation

class TodoApp(App):
    def build(self):
        self.tasks = []
        self.layout = BoxLayout(orientation='vertical')
        self.task_input = TextInput(hint_text='Enter a task', size_hint_y=None, height=40)
        self.layout.add_widget(self.task_input)

        add_button = Button(text='Add Task', size_hint_y=None, height=40)
        add_button.bind(on_press=self.add_task)
        self.layout.add_widget(add_button)

        self.tasks_layout = BoxLayout(orientation='vertical')
        self.layout.add_widget(self.tasks_layout)

        return self.layout

    def add_task(self, instance):
        task_text = self.task_input.text
        if task_text.strip() != '':
            task_label = Label(text=task_text, size_hint_y=None, height=30)
            self.tasks_layout.add_widget(task_label)
            self.tasks.append(task_text)
            self.task_input.text = ''  # Clear input

if __name__ == '__main__':
    TodoApp().run()
