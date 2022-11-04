import pygame
import neat
import os
import random


# neat params - magic numbers
BIRD_PASS_PIPE_REWARD = 5
BIRD_STAY_ALIVE_REWARD = 0.1

# generation count
GENERATION_COUNT = 0


pygame.font.init()
clock = pygame.time.Clock()


GRAV = 1.2
WIN_WIDTH = 550
WIN_HEIGHT = 800
PIPE_DIST = 600

# images form https://github.com/samuelcust/flappy-bird-assets
BIRD_IMG = [pygame.image.load(os.path.join("images", "yellowbird-upflap.png")),
            pygame.image.load(os.path.join("images", "yellowbird-midflap.png")),
            pygame.image.load(os.path.join("images", "yellowbird-downflap.png"))
            ]
BIRD_IMG = list(map(pygame.transform.scale2x, BIRD_IMG))

PIPE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("images", "pipe-green.png")))
BASE_IMG = pygame.image.load(os.path.join("images", "base.png"))
BG_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("images", "background-day.png")))

STAT_FONT = pygame.font.SysFont("Phagspa", 50)


class Bird:
    IMGS = BIRD_IMG
    MAX_ROTATION = 25
    ROT_VEL = 20
    ANIMATION_TIME = 5
    MAX_VEL = 16

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0
        self.vel = 0
        self.height = y
        self.img_count = 0
        self.img = self.IMGS[0]

    def jump(self):
        self.vel = -9
        self.tick_count = 0
        self.height = self.y

    def move(self):
        self.tick_count += 1

        # displacement
        d = self.vel*self.tick_count + GRAV*self.tick_count**2
        if d >= self.MAX_VEL:
            d = self.MAX_VEL
        self.y = self.y + d

        # angle
        if d < 0 or self.y < self.height + 0:
            # tilt up
            self.tilt = self.MAX_ROTATION
        else:
            if self.tilt > -90:
                self.tilt -= self.ROT_VEL

    def draw(self, win):
        # flap the wings
        if self.tilt <= -80:
            self.img = self.IMGS[1]
        else:
            temp_dict = [0, 1, 2, 1]
            self.img_count = (self.img_count + 1) % (self.ANIMATION_TIME*4)
            self.img = self.IMGS[temp_dict[self.img_count // self.ANIMATION_TIME ]]

        # rotate the bird
        rotated_img = pygame.transform.rotate(self.img, self.tilt)
        new_rect = rotated_img.get_rect(center=self.img.get_rect(topleft=(self.x, self.y)).center)

        win.blit(rotated_img, new_rect.topleft)

    def get_mask(self):
        return pygame.mask.from_surface(self.img)

    def reset(self, newx, newy):
        self.x = newx
        self.y = newy


class Pipe:
    GAP = 200
    VEL = 5

    def __init__(self, x):
        self.x = x
        self.height = 0

        self.top = 0
        self.bottom = 0
        self.PIPE_TOP = pygame.transform.flip(PIPE_IMG, False, True)
        self.PIPE_BOTTOM = PIPE_IMG

        self.passed = False
        self.get_height()

    def get_height(self):
        self.height = random.randrange(10, WIN_HEIGHT - self.GAP - 100, 1)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP

    def move(self):
        self.x -= self.VEL

    def draw(self, win):
        win.blit(self.PIPE_TOP, (self.x, self.top))
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    def collide(self, bird):
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)

        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        b_point = bird_mask.overlap(bottom_mask, bottom_offset)
        t_point = bird_mask.overlap(top_mask, top_offset)

        if t_point or b_point:
            return True
        else:
            return False


class Base:
    VEL = 5
    WIDTH = BASE_IMG.get_width()
    HEIGHT = BASE_IMG.get_height()
    IMG = BASE_IMG

    def __init__(self):
        self.top = WIN_HEIGHT - self.HEIGHT + 40
        self.x1 = 0
        self.x2 = self.WIDTH
        self.x3 = self.WIDTH * 2

    def move(self):
        self.x1 -= self.VEL
        self.x2 -= self.VEL
        self.x3 -= self.VEL

        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x3 + self.WIDTH

        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

        if self.x3 + self.WIDTH < 0:
            self.x3 = self.x2 + self.WIDTH

    def draw(self, win):
        win.blit(self.IMG, (self.x1, self.top))
        win.blit(self.IMG, (self.x2, self.top))
        win.blit(self.IMG, (self.x3, self.top))


def draw_window(win, birds, pipes, base, score, gen, alive_count):
    win.blit(BG_IMG, (0, 0))
    for pipe in pipes:
        pipe.draw(win)

    text = STAT_FONT.render("Score: " + str(score), True, (255, 255, 255))
    win.blit(text, (WIN_WIDTH - 10 - text.get_width(), 10))

    text = STAT_FONT.render("Gen: " + str(gen), True, (255, 255, 255))
    win.blit(text, (10, 10))

    text = STAT_FONT.render("Alive: " + str(alive_count), True, (255, 255, 255))
    win.blit(text, (10, 20 + text.get_height()))

    base.draw(win)
    for bird in birds:
        bird.draw(win)

    pygame.display.update()


def ai_main(genomes, config):
    global GENERATION_COUNT
    GENERATION_COUNT += 1

    nets = []
    ge = []
    birds = []

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        birds.append(Bird(150, 200))
        g.fitness = 0
        ge.append(g)

    base = Base()
    pipes = [Pipe(PIPE_DIST)]
    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))

    score = 0
    run = True
    while run:
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # determine the nearest pipe
        pipe_ind = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
                pipe_ind = 1

        # no bird left
        else:
            run = False
            break

        for i, bird in enumerate(birds):
            bird.move()
            ge[i].fitness += BIRD_STAY_ALIVE_REWARD

            # evaluate if bird should jump
            output = nets[i].activate((bird.y, abs(bird.y - pipes[pipe_ind].height),
                                       abs(bird.y - pipes[pipe_ind].bottom)))

            if output[0] > 0.5:
                bird.jump()

        # moving pipes
        rem = []
        add_pipe = False
        for pipe in pipes:
            for i, bird in enumerate(birds):
                # collisions with pipes TODO change for ai
                if pipe.collide(bird):
                    # don't favor birds who make it into the pipe but slam it after
                    ge[i].fitness -= 1
                    birds.pop(i)
                    nets.pop(i)
                    ge.pop(i)

                # can be outside of bird loop ? TODO
                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    add_pipe = True

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)

            pipe.move()

        # some pipe has been passed
        if add_pipe:
            score += 1
            for g in ge:
                # add 1 to the fitness of all birds which passed this pipe
                g.fitness += BIRD_PASS_PIPE_REWARD

            pipes.append(Pipe(PIPE_DIST))

        # removing pipes off screen
        for pipe in rem:
            pipes.remove(pipe)

        for i, bird in enumerate(birds):
            # bird collisions with ground and top of the screen
            if bird.y + bird.img.get_height() > base.top or bird.y < 0:
                birds.pop(i)
                nets.pop(i)
                ge.pop(i)

        base.move()
        draw_window(win, birds, pipes, base, score, GENERATION_COUNT, len(birds))


def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)

    pop = neat.Population(config)

    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    winner = pop.run(ai_main, 50)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")
    run(config_path)

