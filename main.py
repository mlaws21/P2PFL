import pygame
import sys
from run import run_recieve, run_add

def add_model(model_name, port, boot_ip):
    # Print the model name, port, and boot IP data
    # print(f"Model Name: {model_name}, Port: {port}, Boot IP: {boot_ip}")
    run_add(port, boot_ip, model_name)

def receive_model(port, boot_ip):
    # print(f"Receiving Model with Port: {port}, Boot IP: {boot_ip}")
    run_recieve(port, boot_ip)
    

def main():
    # Initialize pygame
    pygame.init()

    # Screen dimensions
    WIDTH, HEIGHT = 500, 500
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("P2P Federated Learning")

    # Colors
    BLUE = (70, 130, 180)
    WHITE = (255, 255, 255)
    DARK_GRAY = (50, 50, 50)
    LIGHT_GRAY = (180, 180, 180)

    # Fonts
    header_font = pygame.font.Font(pygame.font.match_font('arial'), 50)
    button_font = pygame.font.Font(pygame.font.match_font('arial'), 25)
    text_font = pygame.font.Font(pygame.font.match_font('arial'), 20)

    # Button properties
    button_width, button_height = 180, 60
    add_button_rect = pygame.Rect((WIDTH // 2 - button_width // 2, HEIGHT // 2 + 40), (button_width, button_height))
    receive_button_rect = pygame.Rect((WIDTH // 2 - button_width // 2, HEIGHT // 2 + 120), (button_width, button_height))

    # Text input boxes
    port_box = pygame.Rect(WIDTH // 2 - 60, HEIGHT // 2 - 130, 120, 30)
    boot_ip_box = pygame.Rect(WIDTH // 2 - 100, HEIGHT // 2 - 80, 200, 30)
    input_box = pygame.Rect(WIDTH // 2 - 100, HEIGHT // 2 - 30, 200, 40)

    text_input = ""
    port_input = ""
    boot_ip_input = ""
    active_box = None
    cursor_visible = True
    clock = pygame.time.Clock()
    cursor_timer = 0

    # Game loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                if add_button_rect.collidepoint(event.pos):
                    add_model(text_input, port_input, boot_ip_input)
                elif receive_button_rect.collidepoint(event.pos):
                    receive_model(port_input, boot_ip_input)
                elif input_box.collidepoint(event.pos):
                    active_box = "input"
                elif port_box.collidepoint(event.pos):
                    active_box = "port"
                elif boot_ip_box.collidepoint(event.pos):
                    active_box = "boot_ip"
                else:
                    active_box = None

            if event.type == pygame.KEYDOWN and active_box:
                if event.key == pygame.K_RETURN:
                    if active_box == "input":
                        add_model(text_input, port_input, boot_ip_input)
                    elif active_box == "port":
                        print(f"Port entered: {port_input}")
                    elif active_box == "boot_ip":
                        print(f"Boot IP entered: {boot_ip_input}")
                elif event.key == pygame.K_BACKSPACE:
                    if active_box == "input":
                        text_input = text_input[:-1]
                    elif active_box == "port":
                        port_input = port_input[:-1]
                    elif active_box == "boot_ip":
                        boot_ip_input = boot_ip_input[:-1]
                else:
                    if active_box == "input":
                        text_input += event.unicode
                    elif active_box == "port":
                        port_input += event.unicode
                    elif active_box == "boot_ip":
                        boot_ip_input += event.unicode

        # Update cursor visibility
        cursor_timer += clock.get_time()
        if cursor_timer >= 500:  # Toggle every 500 ms
            cursor_visible = not cursor_visible
            cursor_timer = 0

        # Draw background
        screen.fill(BLUE)

        # Draw header text
        header_text = header_font.render("Welcome to P2PFL", True, WHITE)
        screen.blit(header_text, (WIDTH // 2 - header_text.get_width() // 2, 20))

        # Draw labels for text boxes
        port_label = text_font.render("Port #:", True, WHITE)
        boot_ip_label = text_font.render("Boot Address:", True, WHITE)
        model_name_label = text_font.render("Model Name:", True, WHITE)
        screen.blit(port_label, (port_box.x - 65, port_box.y + 5))
        screen.blit(boot_ip_label, (boot_ip_box.x - 130, boot_ip_box.y + 5))
        screen.blit(model_name_label, (input_box.x - 125, input_box.y + 10))

        # Draw Add Model button
        pygame.draw.rect(screen, LIGHT_GRAY, add_button_rect, border_radius=10)
        add_text = button_font.render("Add Model", True, DARK_GRAY)
        screen.blit(add_text, (add_button_rect.x + (button_width - add_text.get_width()) // 2, add_button_rect.y + (button_height - add_text.get_height()) // 2))

        # Draw Receive Model button
        pygame.draw.rect(screen, LIGHT_GRAY, receive_button_rect, border_radius=10)
        receive_text = button_font.render("Receive Model", True, DARK_GRAY)
        screen.blit(receive_text, (receive_button_rect.x + (button_width - receive_text.get_width()) // 2, receive_button_rect.y + (button_height - receive_text.get_height()) // 2))

        # Draw input boxes
        pygame.draw.rect(screen, WHITE, input_box, border_radius=5)
        pygame.draw.rect(screen, WHITE, port_box, border_radius=5)
        pygame.draw.rect(screen, WHITE, boot_ip_box, border_radius=5)

        # Render input text
        input_text_render = text_font.render(text_input, True, DARK_GRAY)
        port_text_render = text_font.render(port_input, True, DARK_GRAY)
        boot_ip_text_render = text_font.render(boot_ip_input, True, DARK_GRAY)
        
        screen.blit(input_text_render, (input_box.x + 10, input_box.y + 10))
        screen.blit(port_text_render, (port_box.x + 10, port_box.y + 5))
        screen.blit(boot_ip_text_render, (boot_ip_box.x + 10, boot_ip_box.y + 5))

        # Draw flashing cursor
        if cursor_visible:
            if active_box == "input":
                cursor_x = input_box.x + 10 + text_font.size(text_input)[0]
                pygame.draw.line(screen, DARK_GRAY, (cursor_x, input_box.y + 10), (cursor_x, input_box.y + 30), 2)
            elif active_box == "port":
                cursor_x = port_box.x + 10 + text_font.size(port_input)[0]
                pygame.draw.line(screen, DARK_GRAY, (cursor_x, port_box.y + 5), (cursor_x, port_box.y + 25), 2)
            elif active_box == "boot_ip":
                cursor_x = boot_ip_box.x + 10 + text_font.size(boot_ip_input)[0]
                pygame.draw.line(screen, DARK_GRAY, (cursor_x, boot_ip_box.y + 5), (cursor_x, boot_ip_box.y + 25), 2)

        # Update display
        pygame.display.flip()

        # Tick clock
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
